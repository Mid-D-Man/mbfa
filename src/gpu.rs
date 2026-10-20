//! Optional GPU backend for the match table (cargo feature `gpu`).
//!
//! Compiled only with `--features gpu` (wgpu 30, Rust 1.87+). At run time it is
//! used only when
//!
//! * the acceleration policy allows it (`matchtable::Accel::Auto` or `Force`),
//! * an adapter passes the probe: a real GPU (software adapters are refused
//!   unless `allow_software_adapter(true)`), enough buffer and workgroup
//!   limits, and a **self-test** whose table matches the CPU reference bit for
//!   bit, and
//! * the input is large enough to be worth the transfer (`ACCEL_MIN_BYTES`).
//!
//! Anything else, including any wgpu error at run time, returns `None` and the
//! caller uses the CPU hash chains. The compressed output is identical either
//! way; the GPU only changes how fast the table is built.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::matchtable::{compute_table_cpu, Params, PrevIndex, Table, COMPARE_CAP};

const SHADER: &str = include_str!("gpu/match.wgsl");
const WORKGROUP: u32 = 64;
/// Positions per dispatch: short enough to stay far below GPU watchdog limits.
const CHUNK: usize = 1 << 18;
/// Smallest `max_storage_buffer_binding_size` worth using.
const MIN_BINDING: u64 = 64 << 20;

static ALLOW_SOFTWARE: AtomicBool = AtomicBool::new(false);
static TABLES_BUILT: AtomicUsize = AtomicUsize::new(0);

/// How many match tables the GPU has built in this process (for tests and
/// benchmarks that want to know the GPU path really ran).
pub fn tables_built() -> usize {
    TABLES_BUILT.load(Ordering::Relaxed)
}

/// Allow software (CPU) Vulkan adapters such as lavapipe. For tests only: they
/// are far slower than the plain CPU path.
pub fn allow_software_adapter(allow: bool) {
    ALLOW_SOFTWARE.store(allow, Ordering::Relaxed);
}

/// What the probe found.
#[derive(Clone, Debug)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    pub device_type: String,
}

/// Wall-clock split of the most recent `compute_table` call, for benchmarks.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub upload_ms: f64,
    pub kernel_and_readback_ms: f64,
    pub chunks: usize,
}

struct Resident {
    idx: Arc<PrevIndex>,
    input_buf: wgpu::Buffer,
    prev_buf: wgpu::Buffer,
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    summary: AdapterSummary,
    max_binding: u64,
    broken: Arc<AtomicBool>,
    resident: Mutex<Option<Resident>>,
    timing: Mutex<Timing>,
    /// One dispatch sequence at a time (the two parallel scans may both ask).
    busy: Mutex<()>,
}

static CONTEXT: Mutex<Option<Arc<GpuContext>>> = Mutex::new(None);
static PROBE_LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Why each adapter was accepted or rejected in the most recent probe (empty if
/// the machine reported no adapters at all). For diagnostics.
pub fn probe_report() -> Vec<String> {
    PROBE_LOG.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn log_probe(line: String) {
    PROBE_LOG.lock().unwrap_or_else(|e| e.into_inner()).push(line);
}
static PROBED_WITH: Mutex<Option<bool>> = Mutex::new(None);

/// The GPU context, probing the machine on first use. `None` if there is no
/// usable adapter (the normal case on a server or in CI). The probe is redone
/// if `allow_software_adapter` changed since the last one.
pub fn context() -> Option<Arc<GpuContext>> {
    let allow = ALLOW_SOFTWARE.load(Ordering::Relaxed);
    let mut probed = PROBED_WITH.lock().unwrap_or_else(|e| e.into_inner());
    let mut slot = CONTEXT.lock().unwrap_or_else(|e| e.into_inner());
    if *probed != Some(allow) {
        *slot = GpuContext::probe(allow).map(Arc::new);
        *probed = Some(allow);
    }
    match slot.as_ref() {
        Some(c) if !c.broken.load(Ordering::Relaxed) => Some(Arc::clone(c)),
        _ => None,
    }
}

/// Description of the adapter that would be used, or `None`.
pub fn describe() -> Option<AdapterSummary> {
    context().map(|c| c.summary.clone())
}

impl GpuContext {
    pub fn summary(&self) -> &AdapterSummary { &self.summary }

    /// Timing of the last `compute_table` call.
    pub fn last_timing(&self) -> Timing {
        *self.timing.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn probe(allow_software: bool) -> Option<GpuContext> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        PROBE_LOG.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let mut adapters: Vec<wgpu::Adapter> =
            pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY));
        if adapters.is_empty() { log_probe("no adapters reported by Vulkan / Metal / DirectX 12".into()); }
        adapters.retain(|a| {
            let info = a.get_info();
            let l = a.limits();
            let why = if !allow_software && info.device_type == wgpu::DeviceType::Cpu {
                Some("software adapter (not used unless allowed)".to_string())
            } else if (l.max_storage_buffer_binding_size as u64) < MIN_BINDING {
                Some(format!("storage buffer limit {} B is below {} B", l.max_storage_buffer_binding_size, MIN_BINDING))
            } else if l.max_storage_buffers_per_shader_stage < 3
                || l.max_compute_invocations_per_workgroup < WORKGROUP
                || l.max_compute_workgroup_size_x < WORKGROUP
                || l.max_compute_workgroups_per_dimension < (CHUNK as u32 / WORKGROUP)
            {
                Some("compute limits too small".to_string())
            } else {
                None
            };
            match why {
                Some(w) => { log_probe(format!("{} ({:?}): rejected, {}", info.name, info.backend, w)); false }
                None => true,
            }
        });
        let rank = |t: wgpu::DeviceType| match t {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Other => 3,
            wgpu::DeviceType::Cpu => 4,
        };
        adapters.sort_by_key(|a| rank(a.get_info().device_type));

        for adapter in adapters {
            let info = adapter.get_info();
            match Self::open(&adapter) {
                None => log_probe(format!("{} ({:?}): could not create a device or pipeline", info.name, info.backend)),
                Some(ctx) => {
                    if ctx.self_test() {
                        log_probe(format!("{} ({:?}): accepted", info.name, info.backend));
                        return Some(ctx);
                    }
                    log_probe(format!("{} ({:?}): rejected, self-test did not match the CPU reference", info.name, info.backend));
                }
            }
        }
        None
    }

    fn open(adapter: &wgpu::Adapter) -> Option<GpuContext> {
        let info = adapter.get_info();
        let limits = adapter.limits();
        let max_binding: u64 = (limits.max_storage_buffer_binding_size as u64).min(1 << 30);
        let required = wgpu::Limits {
            max_storage_buffer_binding_size: max_binding,
            max_buffer_size: limits.max_buffer_size.min(1 << 30),
            ..wgpu::Limits::downlevel_defaults()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mbfa match table"),
            required_features: wgpu::Features::empty(),
            required_limits: required,
            ..Default::default()
        }))
        .ok()?;

        let broken = Arc::new(AtomicBool::new(false));
        {
            let b = Arc::clone(&broken);
            device.on_uncaptured_error(Arc::new(move |_err| {
                b.store(true, Ordering::Relaxed);
            }));
        }

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mbfa match.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mbfa match layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, false),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mbfa match pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mbfa match"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if broken.load(Ordering::Relaxed) { return None; }

        Some(GpuContext {
            device,
            queue,
            pipeline,
            layout,
            summary: AdapterSummary {
                name: info.name.clone(),
                backend: format!("{:?}", info.backend),
                device_type: format!("{:?}", info.device_type),
            },
            max_binding,
            broken,
            resident: Mutex::new(None),
            timing: Mutex::new(Timing::default()),
            busy: Mutex::new(()),
        })
    }

    /// The probe's last gate: the table for a small synthetic buffer must equal
    /// the CPU reference exactly, for several parameter sets. A driver that
    /// computes anything else is never used.
    fn self_test(&self) -> bool {
        let mut x: u64 = 0x1234_5678_9ABC_DEF1;
        let mut next = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let mut data: Vec<u8> = Vec::new();
        while data.len() < 24_000 {
            match next() % 4 {
                0 => { let b = next() as u8; for _ in 0..(1 + next() % 40) { data.push(b); } }
                1 => { for _ in 0..(1 + next() % 30) { data.push((next() % 6) as u8); } }
                2 if data.len() > 100 => {
                    let s = (next() as usize) % (data.len() - 50);
                    let l = 1 + (next() as usize) % 40;
                    let blk = data[s..s + l].to_vec();
                    data.extend(blk);
                }
                _ => { for _ in 0..(1 + next() % 20) { data.push(next() as u8); } }
            }
        }
        data.extend(std::iter::repeat(0u8).take(COMPARE_CAP + 300)); // exercises the overflow marker
        let idx = Arc::new(PrevIndex::build(&data));
        for &(start_at, max_off, max_len, chain_limit) in &[
            (0usize, (1usize << 12) - 1, 255usize, 64usize),
            (0, (1 << 16) - 1, 4095, 256),
            (300, (1 << 9) - 1, 255, 16),
        ] {
            let p = Params { start_at, max_off, max_len, chain_limit };
            let want = compute_table_cpu(&data, &idx, &p);
            match self.compute_table(&data, &idx, &p) {
                Some(got) if got.start == want.start && got.entries == want.entries => {}
                _ => return false,
            }
        }
        true
    }

    /// Build the match table for `input` on the GPU, or `None` (the caller falls
    /// back to the CPU) on any problem.
    pub fn compute_table(&self, input: &[u8], idx: &Arc<PrevIndex>, p: &Params) -> Option<Table> {
        if self.broken.load(Ordering::Relaxed) { return None; }
        let n = input.len();
        let end = n.saturating_sub(2);
        let start = p.start_at.min(end);
        if end == start { return Some(Table { start, entries: Vec::new() }); }
        if (n as u64) * 4 > self.max_binding || n > u32::MAX as usize - 8 { return None; }

        let _guard = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        let t_upload = Instant::now();

        // Resident buffers: the same input is scanned several times with different
        // settings, so it is uploaded once.
        let mut resident = self.resident.lock().unwrap_or_else(|e| e.into_inner());
        let reuse = matches!(resident.as_ref(), Some(r) if Arc::ptr_eq(&r.idx, idx));
        if !reuse {
            let mut words = vec![0u32; n / 4 + 2];
            for (k, b) in input.iter().enumerate() {
                words[k >> 2] |= (*b as u32) << ((k & 3) * 8);
            }
            let input_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mbfa input"),
                size: (words.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&input_buf, 0, bytemuck::cast_slice(&words));
            let prev_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mbfa prev"),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&prev_buf, 0, bytemuck::cast_slice(&idx.prev[..n]));
            *resident = Some(Resident { idx: Arc::clone(idx), input_buf, prev_buf });
        }
        let r = resident.as_ref()?;

        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mbfa params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_size = (CHUNK * 8) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mbfa out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mbfa staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mbfa bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: r.input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: r.prev_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            ],
        });
        let upload_ms = t_upload.elapsed().as_secs_f64() * 1e3;

        let t_run = Instant::now();
        let mut entries: Vec<u32> = Vec::with_capacity((end - start) * 2);
        let mut chunks = 0usize;
        let mut base = start;
        while base < end {
            let count = CHUNK.min(end - base);
            let params: [u32; 8] = [
                n as u32,
                base as u32,
                count as u32,
                p.max_off.min(u32::MAX as usize) as u32,
                p.max_len.min(u32::MAX as usize) as u32,
                p.chain_limit as u32,
                COMPARE_CAP as u32,
                0,
            ];
            self.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&params));
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mbfa match") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mbfa match"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups((count as u32 + WORKGROUP - 1) / WORKGROUP, 1, 1);
            }
            enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (count * 8) as u64);
            self.queue.submit(Some(enc.finish()));

            let slice = staging.slice(0..(count * 8) as u64);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            if self.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
                self.broken.store(true, Ordering::Relaxed);
                return None;
            }
            match rx.recv() {
                Ok(Ok(())) => {}
                _ => {
                    self.broken.store(true, Ordering::Relaxed);
                    return None;
                }
            }
            {
                let view = match slice.get_mapped_range() {
                    Ok(v) => v,
                    Err(_) => {
                        self.broken.store(true, Ordering::Relaxed);
                        return None;
                    }
                };
                let words: &[u32] = bytemuck::cast_slice(&view);
                entries.extend_from_slice(&words[..count * 2]);
            }
            staging.unmap();
            if self.broken.load(Ordering::Relaxed) { return None; }
            base += count;
            chunks += 1;
        }
        let kernel_ms = t_run.elapsed().as_secs_f64() * 1e3;
        *self.timing.lock().unwrap_or_else(|e| e.into_inner()) =
            Timing { upload_ms, kernel_and_readback_ms: kernel_ms, chunks };
        TABLES_BUILT.fetch_add(1, Ordering::Relaxed);
        Some(Table { start, entries })
    }
}
