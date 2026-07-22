# scripts/gen_special_files.py
#!/usr/bin/env python3
"""Generate MBFA special benchmark files."""

import argparse
import hashlib
import json
import math
import os
import random
import struct


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="bench_files_special")
    return p.parse_args()


def emit(out_dir, filename, data, description, manifest):
    path = os.path.join(out_dir, filename)
    if isinstance(data, str):
        data = data.encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    size = len(data)
    print(f"  {filename:<42} {size:>9,} bytes  — {description}")
    manifest.append({"file": filename, "size": size, "description": description})


# ── Binary STL sphere (5000 triangles ≈ 250 KB) ───────────────────────────────

def gen_stl_sphere(n_lat=50, n_lon=50):
    triangles = []
    for i in range(n_lat):
        for j in range(n_lon):
            phi0 = math.pi * i / n_lat
            phi1 = math.pi * (i + 1) / n_lat
            th0  = 2 * math.pi * j / n_lon
            th1  = 2 * math.pi * (j + 1) / n_lon

            def sph(phi, th):
                return (math.sin(phi) * math.cos(th),
                        math.cos(phi),
                        math.sin(phi) * math.sin(th))

            v00, v01 = sph(phi0, th0), sph(phi0, th1)
            v10, v11 = sph(phi1, th0), sph(phi1, th1)
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    n_tris = len(triangles)
    buf = bytearray(80)
    buf += struct.pack("<I", n_tris)
    for v0, v1, v2 in triangles:
        nx = (v0[0] + v1[0] + v2[0]) / 3
        ny = (v0[1] + v1[1] + v2[1]) / 3
        nz = (v0[2] + v1[2] + v2[2]) / 3
        nl = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        buf += struct.pack("<fff", nx / nl, ny / nl, nz / nl)
        buf += struct.pack("<fff", *v0)
        buf += struct.pack("<fff", *v1)
        buf += struct.pack("<fff", *v2)
        buf += struct.pack("<H", 0)
    assert len(buf) == 84 + n_tris * 50
    return bytes(buf)


# ── Binary PLY heightmap grid ─────────────────────────────────────────────────

def gen_ply_heightmap(grid_w=51, grid_h=51):
    FREQ_X = 7
    FREQ_Y = 5

    vertices = []
    for iy in range(grid_h):
        for ix in range(grid_w):
            sx = ix / max(grid_w - 1, 1)
            sy = iy / max(grid_h - 1, 1)
            x  = sx * 10.0 - 5.0
            y  = sy * 10.0 - 5.0
            z  = 2.0 * math.sin(sx * FREQ_X * math.pi * 2) * math.cos(sy * FREQ_Y * math.pi * 2)
            dz_dx = (2.0 * FREQ_X * math.pi * 2
                     * math.cos(sx * FREQ_X * math.pi * 2)
                     * math.cos(sy * FREQ_Y * math.pi * 2)) / 10.0
            dz_dy = (-2.0 * FREQ_Y * math.pi * 2
                     * math.sin(sx * FREQ_X * math.pi * 2)
                     * math.sin(sy * FREQ_Y * math.pi * 2)) / 10.0
            nx, ny, nz = -dz_dx, -dz_dy, 1.0
            nl = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx / nl, ny / nl, nz / nl
            vertices.append((x, y, z, nx, ny, nz, sx, sy))

    n_verts = len(vertices)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float u\nproperty float v\n"
        "end_header\n"
    ).encode()
    data = bytearray(header)
    for v in vertices:
        data += struct.pack("<8f", *v)
    return bytes(data)


# ── GLB (unchanged) ───────────────────────────────────────────────────────────

def gen_glb():
    rng = random.Random(42)
    verts = []
    for _ in range(3000):
        phi   = rng.uniform(0, math.pi)
        theta = rng.uniform(0, 2 * math.pi)
        r     = rng.uniform(0.5, 1.5)
        verts.extend([r * math.sin(phi) * math.cos(theta),
                      r * math.cos(phi),
                      r * math.sin(phi) * math.sin(theta)])

    idx_raw  = struct.pack("<3H", 0, 1, 2)
    idx_pad  = (4 - len(idx_raw) % 4) % 4
    idx_data = idx_raw + b"\x00" * idx_pad
    vert_data = struct.pack(f"<{len(verts)}f", *verts)
    bin_data  = idx_data + vert_data
    bin_pad   = (4 - len(bin_data) % 4) % 4
    bin_data += b"\x00" * bin_pad

    gltf = json.dumps({
        "asset": {"version": "2.0", "generator": "MBFA-bench"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 1}, "indices": 0}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": len(verts) // 3,
             "type": "VEC3", "min": [-1.5, -1.5, -1.5], "max": [1.5, 1.5, 1.5]},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0,
             "byteLength": len(idx_data), "target": 34963},
            {"buffer": 0, "byteOffset": len(idx_data),
             "byteLength": len(vert_data), "target": 34962},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }, separators=(",", ":")).encode()
    json_pad  = (4 - len(gltf) % 4) % 4
    gltf     += b" " * json_pad

    json_chunk = struct.pack("<II", len(gltf),     0x4E4F534A) + gltf
    bin_chunk  = struct.pack("<II", len(bin_data), 0x004E4942) + bin_data
    total_len  = 12 + len(json_chunk) + len(bin_chunk)
    return struct.pack("<III", 0x46546C67, 2, total_len) + json_chunk + bin_chunk


# ── Row-periodic terrain heightmap ────────────────────────────────────────────

def gen_terrain_raw(width, height, seed=1):
    row_bytes = width * 2
    period    = max(4, 65534 // row_bytes)

    FREQ_X    = 7
    AMPLITUDE = 0.35
    BASE      = 0.50

    base_rows = []
    for y_base in range(period):
        sy       = y_base / max(period - 1, 1)
        y_offset = 0.06 * math.sin(sy * math.pi * 2 + 1.0)
        row = []
        for x in range(width):
            sx = x / max(width - 1, 1)
            h_primary = (BASE
                         + AMPLITUDE * math.sin(sx * FREQ_X * math.pi * 2)
                         + 0.10 * math.cos(sx * FREQ_X * 2 * math.pi * 2)
                         + 0.05 * math.sin(sx * FREQ_X * 3 * math.pi * 2)
                         + y_offset)
            h_noise = 0.08 * math.sin(sx * 41 * math.pi * 2 + seed * 0.7)
            h = h_primary + h_noise
            v = max(0, min(65535, int(h * 65535)))
            row.append(v)
        base_rows.append(row)

    data = bytearray(width * height * 2)
    for y in range(height):
        src       = base_rows[y % period]
        row_start = y * width * 2
        for x in range(width):
            v = src[x]
            data[row_start + x * 2]     = v & 0xFF
            data[row_start + x * 2 + 1] = (v >> 8) & 0xFF

    return bytes(data)


# ── Structured PE/COFF DLL ────────────────────────────────────────────────────

def gen_minimal_dll():
    FUNC_SIZE  = 64
    CODE_OFF   = 0x200
    N_FUNCS    = (65536 - CODE_OFF) // FUNC_SIZE
    N_UTILITY  = 16
    CALL_PROB  = 0.45

    PROLOGUE = bytes([
        0x55,
        0x48, 0x89, 0xE5,
        0x41, 0x54,
        0x41, 0x55,
        0x48, 0x83, 0xEC, 0x30,
    ])
    EPILOGUE = bytes([
        0x48, 0x83, 0xC4, 0x30,
        0x41, 0x5D,
        0x41, 0x5C,
        0x5D,
        0xC3,
    ])

    rng = random.Random(0xABCD1234)
    code = bytearray()
    func_file_offs = []

    for fn_idx in range(N_FUNCS):
        fn_off = CODE_OFF + fn_idx * FUNC_SIZE
        func_file_offs.append(fn_off)

        fn        = bytearray(PROLOGUE)
        body_left = FUNC_SIZE - len(PROLOGUE) - len(EPILOGUE)

        while body_left >= 5:
            if fn_idx >= N_UTILITY and rng.random() < CALL_PROB:
                target    = rng.randint(0, N_UTILITY - 1)
                call_site = fn_off + len(fn) + 5
                rel       = func_file_offs[target] - call_site
                rel_b     = struct.pack('<i', rel)
                if rel_b[3] in (0x00, 0xFF):
                    fn       += bytes([0xE8]) + rel_b
                    body_left -= 5
                    continue
            fn       += bytes([0x90])
            body_left -= 1

        while len(fn) + len(EPILOGUE) < FUNC_SIZE:
            fn += bytes([0x90])

        fn += EPILOGUE
        assert len(fn) == FUNC_SIZE
        code += fn

    assert len(code) == N_FUNCS * FUNC_SIZE == 65536 - CODE_OFF

    header = bytearray(CODE_OFF)
    header[0] = ord('M')
    header[1] = ord('Z')
    struct.pack_into('<I', header, 0x3C, 0x40)
    header[0x40] = ord('P')
    header[0x41] = ord('E')
    header[0x42] = 0x00
    header[0x43] = 0x00
    struct.pack_into('<H', header, 0x44, 0x8664)
    struct.pack_into('<H', header, 0x46, 1)
    struct.pack_into('<H', header, 0x50, 0x2022)

    result = bytes(header) + bytes(code)
    assert len(result) == 65536
    return result


# ── Real DixScript binary format (.mdix compiled) ─────────────────────────────
#
# Rewritten July 2026 alongside the DixScript dictionary split (see
# mbfa/src/dictionary/dixscript_binary.rs's doc comment). The previous
# version of this function wrapped every section in a generic Object tag
# (0x09, [Count:4][Key-Value pairs...]) with made-up keys like
# "max_folds"/"offset_bits_min" that don't exist anywhere in real DixScript
# -- those are MBFA's own config names, not DixScript's. The real writers
# (DixScript-Rust's BinarySerialization/SectionWriters/*.rs) never use the
# generic Object wrapper for @CONFIG/@DATA/@SECURITY at the section level;
# each section has its own dedicated layout, checked directly against:
#   - config_section_writer.rs   (@CONFIG: real ConfigKey vocabulary)
#   - data_section_writer.rs     (@DATA: DataEntryType::SimpleProperty)
#   - security_section_writer.rs (@SECURITY: block-key -> field list)
#   - value_encoder.rs           (tag bytes: 0x01 Int32 … 0x09 Object)
#   - binary_format.rs           (SectionId, 16-byte header layout)
#   - checksum_validator.rs      (trailing 32-byte SHA-256, no marker)

TAG_INT32  = 0x01
TAG_FLOAT32 = 0x03
TAG_STRING = 0x05
TAG_BOOL   = 0x06

def _enc_int32(v):
    return bytes([TAG_INT32]) + struct.pack('<i', int(v))

def _enc_float32(v):
    return bytes([TAG_FLOAT32]) + struct.pack('<f', float(v))

def _enc_string(s):
    b = s.encode('utf-8')
    return bytes([TAG_STRING]) + struct.pack('<i', len(b)) + b

def _enc_bool(v):
    return bytes([TAG_BOOL, 0x01 if v else 0x00])

def _enc_field_name(s):
    # Bare length-prefixed UTF-8, no type tag -- used for keys (ConfigEntry,
    # DataEntry name, SecurityEntry block key, SecurityField key), which are
    # never typed values themselves.
    b = s.encode('utf-8')
    return struct.pack('<i', len(b)) + b

def _section(section_id, entry_count, body):
    # [Section ID: 4][Section Length: 4][Entry Count: 4][Entries...]
    # section_length = end_pos - start_pos, i.e. it includes the id field,
    # the length field itself, the entry count, AND the entry bytes --
    # matches config_section_writer.rs's start_pos-before-id /
    # end_pos-after-entries measurement exactly.
    section_length = 4 + 4 + 4 + len(body)
    return (struct.pack('<I', section_id) + struct.pack('<i', section_length)
            + struct.pack('<i', entry_count) + body)


def gen_binary_dixscript(num_data_entries=400, seed=42):
    rng = random.Random(seed)

    # ── @CONFIG: real ConfigKey vocabulary (others/midx.ebnf), real
    # per-entry format [KeyLen:4][Key][ValueTag:1][ValueData] ────────────────
    config_entries = [
        ("version",             _enc_string("1.0.0")),
        ("encoding",            _enc_string("UTF-8")),
        ("author",              _enc_string("MBFA-CI")),
        ("features",            _enc_string("advanced")),
        ("debug_mode",          _enc_string("regular")),
        ("error_handling",      _enc_string("halt")),
        ("compatibility_mode",  _enc_string("strict")),
    ]
    config_body = b''.join(_enc_field_name(k) + v for k, v in config_entries)
    config_section = _section(1, len(config_entries), config_body)

    # ── @SECURITY: real block-key -> field-list format
    # [BlockKeyLen:4][BlockKey][FieldCount:4][Fields: [KeyLen:4][Key][Value]] ─
    security_blocks = [
        ("encryption", [("mode", _enc_string("keyfile")), ("algorithm", _enc_string("aes256-gcm"))]),
        ("validation", [("level", _enc_string("strict")), ("enabled", _enc_bool(True))]),
        ("keystore",   [("provider", _enc_string("local")), ("rotate_days", _enc_int32(90))]),
    ]
    sec_body = bytearray()
    for block_key, fields in security_blocks:
        sec_body += _enc_field_name(block_key)
        sec_body += struct.pack('<i', len(fields))
        for fk, fv in fields:
            sec_body += _enc_field_name(fk) + fv
    security_section = _section(4, len(security_blocks), bytes(sec_body))

    # ── @DATA: real DataEntryType::SimpleProperty format
    # [Type:1=0x01][NameLen:4][Name][Value] per entry, entries share the
    # section's flat entry list (no per-entry Object wrapper) ───────────────
    ENTRY_NAMES = [
        "Platform", "Encoder", "Decoder", "Archive", "Entropy",
        "Filter", "Compress", "Decompress", "Fold", "Unfold",
        "Config", "Scanner", "Parser", "Writer", "Reader",
    ]
    ENTRY_TYPES = [
        "config", "filter", "encoder", "decoder",
        "archive", "entropy", "pipeline", "transform",
    ]
    TAG_NAMES = [
        "core", "stable", "beta", "experimental",
        "required", "optional", "deprecated",
    ]

    data_body = bytearray()
    for i in range(num_data_entries):
        # Each SimpleProperty entry's "value" here is itself the only
        # payload after [Type][NameLen][Name] -- real DixScript data entries
        # are single name=value pairs, not a props-list per entry (that
        # shape was this function's other main fabrication). To still
        # exercise realistic entry diversity we emit num_data_entries
        # separate SimpleProperty entries cycling through plausible
        # name/value pairs, which is what a real @DATA block with many
        # scalar assignments actually looks like on the wire.
        name = f"{ENTRY_NAMES[i % len(ENTRY_NAMES)]}_{i}"
        choice = i % 4
        if choice == 0:
            value = _enc_int32(rng.randint(0, 1024))
        elif choice == 1:
            value = _enc_string(ENTRY_TYPES[i % len(ENTRY_TYPES)])
        elif choice == 2:
            value = _enc_bool(rng.random() > 0.25)
        else:
            value = _enc_float32(rng.uniform(0.0, 1.0))
        data_body += bytes([0x01]) + _enc_field_name(name) + value

    data_section = _section(3, num_data_entries, bytes(data_body))

    MAGIC_U32    = 0x4D444958
    HEADER_SIZE  = 16

    sections = [
        (1, config_section),
        (3, data_section),
        (4, security_section),
    ]

    layout  = []
    payload = bytearray()
    cur_off = HEADER_SIZE
    for sec_id, sec_bytes in sections:
        layout.append((sec_id, cur_off, len(sec_bytes)))
        payload += sec_bytes
        cur_off += len(sec_bytes)

    offset_table_pos = HEADER_SIZE + len(payload)

    offset_table = bytearray()
    for sec_id, off, length in layout:
        offset_table += struct.pack('<I', sec_id)
        offset_table += struct.pack('<i', off)
        offset_table += struct.pack('<i', length)

    flags = 0x01 | 0x04

    header  = struct.pack('<I', MAGIC_U32)
    header += bytes([1, 0, 0])
    header += bytes([flags])
    header += struct.pack('<i', len(sections))
    header += struct.pack('<i', offset_table_pos)
    assert len(header) == HEADER_SIZE

    binary   = header + bytes(payload) + bytes(offset_table)
    checksum = hashlib.sha256(binary).digest()
    return binary + checksum


# ── MBFA Showcase: repetitive block ──────────────────────────────────────────

def gen_showcase_repetitive(block_size=4096, repeat_count=64, seed=1):
    rng = random.Random(seed)
    block = bytearray()
    for i in range(block_size):
        if i % 32 < 16:
            block.append(i & 0xFF)
        elif i % 32 < 24:
            block.append(rng.randint(0, 15))
        else:
            block.append(0x00)
    return bytes(block) * repeat_count


# ── MBFA Showcase: sparse binary ─────────────────────────────────────────────

def gen_showcase_sparse(total_size=256 * 1024, seed=2):
    rng = random.Random(seed)
    data = bytearray(total_size)

    BLOCK_INTERVAL = 64
    PAYLOAD_LEN    = 28

    for i in range(0, total_size - PAYLOAD_LEN - 4, BLOCK_INTERVAL):
        block_idx = i // BLOCK_INTERVAL

        data[i]   = 0xFF
        data[i+1] = PAYLOAD_LEN
        struct.pack_into('<H', data, i + 2, block_idx & 0xFFFF)

        for j in range(PAYLOAD_LEN):
            if j % 4 == 0:
                val = (block_idx * PAYLOAD_LEN + j) & 0xFF
            elif j % 4 == 1:
                val = ((block_idx * PAYLOAD_LEN + j) >> 8) & 0xFF
            elif j % 4 == 2:
                val = (block_idx % 8) + 1
            else:
                val = [0x00, 0x01, 0x02, 0xFF][block_idx % 4]
            data[i + 4 + j] = val

    return bytes(data)


# ── MBFA Showcase: tile-based game map ───────────────────────────────────────

def gen_showcase_gamemap(width=256, height=256, seed=3):
    rng = random.Random(seed)
    data = bytearray(width * height * 2)

    def set_tile(x, y, tile_type, variant=0):
        if 0 <= x < width and 0 <= y < height:
            pos         = (y * width + x) * 2
            data[pos]   = tile_type
            data[pos+1] = variant & 0xFF

    rooms = []
    for _ in range(24):
        rx = rng.randint(5, width  - 30)
        ry = rng.randint(5, height - 20)
        rw = rng.randint(8, 22)
        rh = rng.randint(6, 16)
        rooms.append((rx, ry, rw, rh))

        for yy in range(ry, ry + rh):
            for xx in range(rx, rx + rw):
                set_tile(xx, yy, 0x01)

        for xx in range(rx - 1, rx + rw + 1):
            set_tile(xx, ry - 1,  0x02)
            set_tile(xx, ry + rh, 0x02)
        for yy in range(ry, ry + rh):
            set_tile(rx - 1,   yy, 0x02)
            set_tile(rx + rw,  yy, 0x02)

    for i in range(len(rooms) - 1):
        r1, r2 = rooms[i], rooms[i + 1]
        cx1 = r1[0] + r1[2] // 2
        cy1 = r1[1] + r1[3] // 2
        cx2 = r2[0] + r2[2] // 2
        cy2 = r2[1] + r2[3] // 2
        for xx in range(min(cx1, cx2), max(cx1, cx2) + 1):
            set_tile(xx, cy1, 0x01)
        for yy in range(min(cy1, cy2), max(cy1, cy2) + 1):
            set_tile(cx2, yy, 0x01)

    for i, (rx, ry, rw, rh) in enumerate(rooms):
        cx = rx + rw // 2
        cy = ry + rh // 2
        set_tile(cx, cy, 0x10 + (i % 3), i & 0xFF)

    return bytes(data)


# ── Text content constants ────────────────────────────────────────────────────

UNITY_CS = """\
using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class GameManager : MonoBehaviour
{
    [Header("Player Settings")]
    public float moveSpeed = 5.0f;
    public float jumpForce = 8.0f;
    public int maxHealth = 100;
    public int currentHealth;

    [Header("References")]
    public Transform playerTransform;
    public Camera mainCamera;
    public AudioSource audioSource;
    public List<GameObject> enemies = new List<GameObject>();

    private Rigidbody rb;
    private bool isGrounded;
    private Vector3 velocity;
    private static GameManager _instance;

    public static GameManager Instance { get { return _instance; } }

    void Awake()
    {
        if (_instance != null && _instance != this) Destroy(this.gameObject);
        else _instance = this;
        DontDestroyOnLoad(this.gameObject);
        currentHealth = maxHealth;
        rb = GetComponent<Rigidbody>();
    }

    void Start() { InitializeGame(); StartCoroutine(SpawnCoroutine()); }
    void Update() { HandleInput(); UpdateUI(); CheckWinCondition(); }
    void FixedUpdate() { MovePlayer(); }

    private void HandleInput()
    {
        float h = Input.GetAxisRaw("Horizontal");
        float v = Input.GetAxisRaw("Vertical");
        velocity = new Vector3(h, 0, v).normalized * moveSpeed;
        if (Input.GetButtonDown("Jump") && isGrounded)
        { rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse); isGrounded = false; }
        if (Input.GetKeyDown(KeyCode.Escape)) PauseGame();
    }

    private void MovePlayer() { rb.MovePosition(rb.position + velocity * Time.fixedDeltaTime); }
    private void InitializeGame() { enemies.Clear(); currentHealth = maxHealth; Time.timeScale = 1f; }

    public void TakeDamage(int amount)
    {
        currentHealth = Mathf.Clamp(currentHealth - amount, 0, maxHealth);
        if (currentHealth <= 0) GameOver();
    }

    public void Heal(int amount) { currentHealth = Mathf.Clamp(currentHealth + amount, 0, maxHealth); }
    private void GameOver() { Time.timeScale = 0f; UnityEngine.SceneManagement.SceneManager.LoadScene("GameOver"); }
    private void PauseGame() { Time.timeScale = Time.timeScale == 0f ? 1f : 0f; }
    private void UpdateUI() {}
    private void CheckWinCondition() { if (enemies.Count == 0) Debug.Log("You Win!"); }

    private IEnumerator SpawnCoroutine()
    { while (true) { yield return new WaitForSeconds(5f); Debug.Log("Spawning enemy"); } }

    void OnCollisionEnter(Collision c)
    {
        if (c.gameObject.CompareTag("Ground")) isGrounded = true;
        if (c.gameObject.CompareTag("Enemy")) TakeDamage(10);
    }

    void OnTriggerEnter(Collider o)
    {
        if (o.CompareTag("HealthPickup")) { Heal(25); Destroy(o.gameObject); }
        if (o.CompareTag("Enemy")) enemies.Remove(o.gameObject);
    }
}
"""

UNITY_SCENE = """\
%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!104 &2
RenderSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 9
  m_Fog: 0
  m_FogColor: {r: 0.5, g: 0.5, b: 0.5, a: 1}
  m_FogMode: 3
  m_FogDensity: 0.01
  m_AmbientSkyColor: {r: 0.212, g: 0.227, b: 0.259, a: 1}
  m_AmbientIntensity: 1
--- !u!1 &100000000
GameObject:
  m_ObjectHideFlags: 0
  serializedVersion: 6
  m_Component:
  - component: {fileID: 100000001}
  - component: {fileID: 100000002}
  m_Layer: 0
  m_Name: Main Camera
  m_TagString: MainCamera
  m_IsActive: 1
--- !u!4 &100000001
Transform:
  m_ObjectHideFlags: 0
  m_LocalPosition: {x: 0, y: 1, z: -10}
  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}
  m_LocalScale: {x: 1, y: 1, z: 1}
--- !u!20 &100000002
Camera:
  m_ObjectHideFlags: 0
  serializedVersion: 2
  m_ClearFlags: 1
  m_BackGroundColor: {r: 0.19, g: 0.3, b: 0.47, a: 0}
  m_NearClipPlane: 0.3
  m_FarClipPlane: 1000
  m_FieldOfView: 60
  m_Orthographic: 0
  m_OrthographicSize: 5
"""

UNITY_SHADER = """\
Shader "Custom/PBR_Terrain"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (1,1,1,1)
        _BaseMap ("Base Map", 2D) = "white" {}
        _NormalMap ("Normal Map", 2D) = "bump" {}
        _Metallic ("Metallic", Range(0,1)) = 0.0
        _Smoothness ("Smoothness", Range(0,1)) = 0.5
        _OcclusionStrength ("Occlusion Strength", Range(0,1)) = 1.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        LOD 300
        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode"="UniversalForward" }
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _ADDITIONAL_LIGHTS
            #pragma multi_compile_fog
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            struct Attributes { float4 posOS:POSITION; float3 normOS:NORMAL; float2 uv:TEXCOORD0; };
            struct Varyings  { float4 posCS:SV_POSITION; float2 uv:TEXCOORD0; float3 posWS:TEXCOORD1; float3 normWS:TEXCOORD2; float fogFactor:TEXCOORD3; };
            TEXTURE2D(_BaseMap); SAMPLER(sampler_BaseMap);
            TEXTURE2D(_NormalMap); SAMPLER(sampler_NormalMap);
            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor; float4 _BaseMap_ST;
                float _Metallic; float _Smoothness; float _OcclusionStrength;
            CBUFFER_END
            Varyings vert(Attributes IN)
            {
                Varyings OUT = (Varyings)0;
                VertexPositionInputs pos = GetVertexPositionInputs(IN.posOS.xyz);
                VertexNormalInputs   nm  = GetVertexNormalInputs(IN.normOS);
                OUT.posCS  = pos.positionCS; OUT.posWS = pos.positionWS;
                OUT.normWS = nm.normalWS;
                OUT.uv     = TRANSFORM_TEX(IN.uv, _BaseMap);
                OUT.fogFactor = ComputeFogFactor(pos.positionCS.z);
                return OUT;
            }
            half4 frag(Varyings IN) : SV_Target
            {
                SurfaceData s = (SurfaceData)0;
                s.albedo     = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv).rgb * _BaseColor.rgb;
                s.metallic   = _Metallic; s.smoothness = _Smoothness;
                InputData id = (InputData)0;
                id.positionWS     = IN.posWS;
                id.normalWS       = normalize(IN.normWS);
                id.viewDirectionWS = GetWorldSpaceNormalizeViewDir(IN.posWS);
                id.fogCoord = IN.fogFactor;
                half4 col = UniversalFragmentPBR(id, s);
                col.rgb = MixFog(col.rgb, IN.fogFactor);
                return col;
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/InternalErrorShader"
}
"""

UNREAL_UPLUGIN = json.dumps({
    "FileVersion": 3, "Version": 1, "VersionName": "1.0",
    "FriendlyName": "MBFA Test Plugin",
    "Description": "Test plugin generated for MBFA special benchmark",
    "Category": "Other", "CreatedBy": "MidManStudio",
    "CreatedByURL": "https://github.com/Mid-D-Man/mbfa",
    "CanContainContent": True, "IsBetaVersion": False, "Installed": False,
    "Modules": [{"Name": "MBFAPlugin", "Type": "Runtime",
                 "LoadingPhase": "Default",
                 "AdditionalDependencies": ["Engine", "CoreUObject"]}],
    "Plugins": [],
}, indent=2)

UNREAL_INI = """\
[/Script/Engine.RendererSettings]
r.DefaultFeature.Bloom=True
r.DefaultFeature.AmbientOcclusion=True
r.DefaultFeature.AutoExposure=True
r.DefaultFeature.AutoExposure.Method=0
r.DefaultFeature.MotionBlur=True
r.DefaultFeature.AntiAliasing=2
r.Shadow.CSM.MaxCascades=4
r.Shadow.MaxResolution=2048
r.Shadow.DistanceScale=1.000000
r.TranslucencyLightingVolumeDim=64
r.RefractionQuality=2
r.SSR.Quality=3
r.SceneColorFormat=4
r.GBuffer=1
r.HZBOcclusion=1
r.EarlyZPass=3
r.AllowStaticLighting=True
r.GenerateMeshDistanceFields=False
r.SeparateTranslucency=True
r.CustomDepth=3
r.bEnableRayTracing=False

[/Script/Engine.AudioSettings]
MaximumConcurrentStreams=32
GlobalMinPitchScale=0.400000
GlobalMaxPitchScale=2.000000

[/Script/Engine.PhysicsSettings]
DefaultGravityZ=-980.000000
DefaultTerminalVelocity=4000.000000
bEnableShapeSharing=False
bEnablePCM=True
bWarnMissingLocks=True
MaxPhysicsDeltaTime=0.033333
bSubstepping=False
MaxSubstepDeltaTime=0.016667
MaxSubsteps=6
InitialAverageFrameRate=0.016667
PhysXTreeRebuildRate=10

[/Script/Engine.GameMapsSettings]
GlobalDefaultGameMode=/Script/Engine.GameModeBase
bUseSplitscreen=False
TwoPlayerSplitscreenLayout=Horizontal
ThreePlayerSplitscreenLayout=FavorTop

[/Script/Engine.GarbageCollectionSettings]
gc.MaxObjectsNotConsideredByGC=1
gc.SizeOfPermanentObjectPool=0
gc.FlushStreamingOnGC=0
gc.NumRetriesBeforeForcingGC=10
gc.AllowParallelGC=1
gc.TimeBetweenPurgingPendingKillObjects=60.0
gc.MaxObjectsInEditor=12582912

[/Script/Engine.NetworkSettings]
n.VerifyPeer=1
NetworkEmulationProfiles=()
"""

K8S_YAML = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mbfa-backend
  namespace: production
  labels:
    app: mbfa-backend
    version: v2.1.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mbfa-backend
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: mbfa-backend
    spec:
      containers:
        - name: mbfa-backend
          image: ghcr.io/mid-d-man/mbfa:2.1.0
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: APP_ENV
              value: production
            - name: LOG_LEVEL
              value: info
            - name: DB_HOST
              valueFrom:
                secretKeyRef:
                  name: mbfa-db-secret
                  key: host
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mbfa-db-secret
                  key: password
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "1000m"
              memory: "1Gi"
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 20
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: mbfa-backend-svc
  namespace: production
spec:
  selector:
    app: mbfa-backend
  ports:
    - name: http
      port: 80
      targetPort: 8080
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mbfa-backend-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mbfa-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mbfa-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - api.mbfa.example.com
      secretName: mbfa-tls
  rules:
    - host: api.mbfa.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: mbfa-backend-svc
                port:
                  number: 80
"""

SERVICES_TOML = """\
[package]
name    = "mbfa-services"
version = "2.1.0"
edition = "2021"

[services.api]
host            = "0.0.0.0"
port            = 8080
workers         = 4
timeout_sec     = 30
max_connections = 1000
keepalive_sec   = 75

[services.metrics]
host    = "0.0.0.0"
port    = 9090
enabled = true
path    = "/metrics"

[services.grpc]
host        = "0.0.0.0"
port        = 50051
max_recv_mb = 64
max_send_mb = 64

[database]
host             = "postgres.internal"
port             = 5432
name             = "mbfa_prod"
pool_min         = 5
pool_max         = 50
idle_timeout_sec = 600
max_lifetime_sec = 1800
connect_timeout  = 5

[database.replica]
host     = "postgres-replica.internal"
port     = 5432
pool_min = 2
pool_max = 20

[cache]
backend    = "redis"
url        = "redis://redis-cluster.internal:6379"
pool       = 20
ttl_sec    = 3600
key_prefix = "mbfa:"

[cache.cluster]
nodes = [
  "redis-0.internal:6379",
  "redis-1.internal:6379",
  "redis-2.internal:6379",
]

[auth]
jwt_algorithm   = "RS256"
token_ttl_sec   = 3600
refresh_ttl_sec = 86400
issuer          = "https://auth.mbfa.example.com"
audience        = "mbfa-api"

[logging]
level  = "info"
format = "json"
output = "stdout"

[telemetry]
enabled       = true
endpoint      = "https://otel.internal:4317"
sampling_rate = 0.1
service_name  = "mbfa-backend"

[feature_flags]
compression_v2    = true
new_encoder       = false
experimental_fold = false
enable_bcj        = true
enable_repref     = true
parallel_folds    = true
entropy_v6        = false

[limits]
max_file_size_mb = 100
max_archive_mb   = 1000
max_folds        = 8
max_concurrent   = 256
rate_limit_rps   = 1000

[[workers]]
name     = "compress"
threads  = 8
queue    = 1000
priority = "high"

[[workers]]
name     = "archive"
threads  = 4
queue    = 500
priority = "normal"

[[workers]]
name     = "cleanup"
threads  = 1
queue    = 100
priority = "low"
cron     = "0 * * * *"
"""

PLATFORM_MDIX = """\
// DixScript platform configuration — MBFA benchmark
// Real, grammar-valid DixScript (others/midx.ebnf) -- the previous version
// of this fixture used `module Platform { const ... fn ... }` pseudo-syntax
// that was never valid DixScript to begin with (no `module`/`fn`/top-level
// `enum`/`config` production exists in the real grammar; the only seven
// top-level sections are @CONFIG/@IMPORTS/@DLM/@ENUMS/@QUICKFUNCS/@DATA/
// @SECURITY). See src/dictionary/dixscript.rs's doc comment for the full
// regression writeup this fixture fix is paired with.

@CONFIG(
  version -> "2.1.0",
  encoding -> "UTF-8",
  created -> 2026-07-19T00:00:00Z,
  features -> "advanced",
  debug_mode -> "regular",
  error_handling -> "halt",
  compatibility_mode -> "strict"
)

@DLM(
  DAuditor.enhanced
)

@ENUMS(
  SimilarityGroup { Source = 0, Markup = 1, Binary = 2, Compressed = 3, Other = 4 }
  FilterFlag { None = 0, Delta1 = 1, Delta2 = 2, Delta3 = 3, Delta4 = 4, Stl = 7, Ply = 8, Bcj = 9 }
)

@QUICKFUNCS(
  ~detectPlatform<string>() {
    let mut result = "Unknown";
    chk: os {
      -> "linux" {
        result = "Linux";
      }
      -> "windows" {
        result = "Windows";
      }
      -> "macos" {
        result = "macOS";
      }
      -> miss {
        result = "Unknown";
      }
    }
    return result;
  }

  ~autoChunkSize<int>(availableMb<int>) {
    let mut chunk = availableMb / 256;
    if: chunk < 1 {
      chunk = 1;
    }
    if: chunk > 8 {
      chunk = 8;
    }
    return chunk * 1024 * 1024;
  }
)

@DATA(
  max_folds<int> = 8
  min_improvement<double> = 0.985
  min_fold_bits<int> = 64

  encoder:
    offset_bits_min<int> = 7, offset_bits_max<int> = 24, offset_bits_default<int> = 15,
    length_bits_min<int> = 8, length_bits_max<int> = 24,
    hash_size<int> = 65536, chain_limit<int> = 256, lazy_short_len<int> = 6, rep_slots<int> = 4

  decoder:
    ring_slots<int> = 4, verify_roundtrip<bool> = true, strict_end_token<bool> = false

  entropy:
    min_bytes<int> = 400, v2_min_bytes<int> = 1000, num_variants<int> = 6, parallel<bool> = true

  archive:
    chunk_min_mb<int> = 1, chunk_max_mb<int> = 8, entropy_threshold<double> = 7.5, similarity_groups<int> = 5

  opcodes::
    "BACKREF", "LIT", "END", "REPREF"
)

@SECURITY(
)
"""


# ── Fake Unreal uasset ────────────────────────────────────────────────────────

def gen_uasset():
    UE4_MAGIC = 0x9E2A83C1
    hdr = struct.pack("<I", UE4_MAGIC)
    hdr += struct.pack("<i", -7)
    hdr += struct.pack("<i", 0)
    hdr += struct.pack("<i", 522)
    hdr += struct.pack("<i", 0)
    hdr += struct.pack("<i", 0)
    pkg = b"StaticMesh\x00"
    hdr += struct.pack("<i", len(pkg)) + pkg
    hdr += struct.pack("<I", 0x8)
    hdr += struct.pack("<iiii", 0, 0, 5, 0)
    hdr += struct.pack("<iiii", 3, 0, 0, 0)
    hdr += bytes(16)
    hdr += struct.pack("<iii", 1, 5, 3)

    rng = random.Random(7)
    mesh = bytearray()
    for _ in range(2000):
        phi = rng.uniform(0, math.pi)
        th  = rng.uniform(0, 2 * math.pi)
        x   = math.sin(phi) * math.cos(th) * 100
        y   = math.cos(phi) * 100
        z   = math.sin(phi) * math.sin(th) * 100
        mesh += struct.pack("<fff", x, y, z)
        mesh += struct.pack("<fff", x / 100, y / 100, z / 100)
        mesh += struct.pack("<ff", phi / math.pi, th / (2 * math.pi))

    for i in range(0, 1998, 3):
        mesh += struct.pack("<HHH", i, i + 1, i + 2)

    result = hdr + bytes(mesh)
    target = 65536
    if len(result) < target:
        result += bytes(target - len(result))
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []

    def out(filename, data, description):
        emit(args.output_dir, filename, data, description, manifest)

    print(f"Generating special benchmark files → {args.output_dir}/\n")

    print("── 3D filter targets ────────────────────────────────────────────")
    out("mesh_sphere.stl",   gen_stl_sphere(50, 50),
        "Binary STL sphere — 5000 tris ≈250KB (field-major+stride-1 flag10)")
    out("mesh_sphere.ply",   gen_ply_heightmap(51, 51),
        "Binary PLY terrain grid — 2601 verts 8-prop float (PLY shuffle+stride-fpv)")
    out("scene.glb",         gen_glb(),
        "GLB binary 3D scene — random float mesh (incompressible passthrough)")

    print("\n── Unity project files ──────────────────────────────────────────")
    out("Scripts.cs",        UNITY_CS * 10,
        "Unity C# MonoBehaviour script (×10 repetitions)")
    out("SampleScene.unity", UNITY_SCENE * 6,
        "Unity YAML scene file (×6 repetitions)")
    print("  Generating terrain heightmaps…")
    out("Terrain.raw",       gen_terrain_raw(513, 513, seed=1),
        "Unity terrain 513×513 16-bit LE — row-periodic (period=63 rows, 64638 bytes)")
    out("Terrain_large.raw", gen_terrain_raw(1025, 1025, seed=2),
        "Unity terrain 1025×1025 16-bit LE — row-periodic (period=31 rows, 63550 bytes) h_noise=0.08")
    out("Shaders.shader",    UNITY_SHADER * 5,
        "Unity URP PBR HLSL shader (×5 repetitions)")

    print("\n── Unreal Engine files ──────────────────────────────────────────")
    out("StaticMesh.uasset", gen_uasset(),
        "Unreal Engine 4 static mesh asset (UE4 magic + structured mesh binary)")
    out("plugins.json",      UNREAL_UPLUGIN,
        "Unreal .uplugin JSON descriptor")
    out("DefaultEngine.ini", UNREAL_INI * 5,
        "Unreal DefaultEngine.ini config (×5 repetitions)")

    print("\n── Config / script formats ──────────────────────────────────────")
    out("k8s_config.yaml",   K8S_YAML * 5,
        "Kubernetes deployment YAML (×5 repetitions)")
    out("services.toml",     SERVICES_TOML * 4,
        "TOML services configuration (×4 repetitions)")
    out("platform.mdix",     PLATFORM_MDIX * 5,
        "DixScript platform config — text source format (×5 repetitions)")

    print("\n── DixScript compiled binary ────────────────────────────────────")
    out("DixScript_compiled.mdix", gen_binary_dixscript(num_data_entries=400),
        "Compiled DixScript binary — MDIX format (magic 0x4D444958 LE, CONFIG+DATA, SHA-256 footer)")

    print("\n── Binary / assembly ────────────────────────────────────────────")
    out("Assembly.dll",      gen_minimal_dll(),
        "PE/COFF DLL — 1016×64-byte function stubs + near CALL/JMP (BCJ filter target)")

    print("\n── MBFA Showcase ────────────────────────────────────────────────")
    out("Showcase_Repetitive.bin", gen_showcase_repetitive(block_size=4096, repeat_count=64),
        "256KB: 4096-byte block ×64 — period fits in ob=13 window → MBFA <1%")
    out("Showcase_Sparse.bin",     gen_showcase_sparse(total_size=256 * 1024),
        "256KB: 50% non-zero structured payload + 50% zero padding, 64-byte period")
    out("Showcase_GameMap.bin",    gen_showcase_gamemap(width=256, height=256),
        "128KB: tile-based 2D map — ~85% empty (0x00), rooms+corridors")

    manifest_path = os.path.join(args.output_dir, "special_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "generated_by": "gen_special_files.py",
            "files": manifest,
            "total_files": len(manifest),
            "total_bytes": sum(m["size"] for m in manifest),
        }, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"Generated {len(manifest)} files, "
          f"{sum(m['size'] for m in manifest):,} bytes total")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
