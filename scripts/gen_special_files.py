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

# ── Video containers ──────────────────────────────────────────────────────────
#
# Real codec output (H.264, VP9, MJPEG scan data) is close to incompressible, so
# the coded-frame payloads below are seeded pseudo-random bytes: a stand-in, not
# real video. What these files model faithfully is everything *around* the
# payload, which is where a general-purpose compressor can still find
# redundancy: box / element headers, per-frame length prefixes, sample tables
# and indexes, parameter sets repeated at every keyframe, JPEG tables repeated
# in every MJPEG frame, MPEG-TS packet headers and continuity counters.
#
# The one exception is Y4M, which is raw YUV and therefore genuinely
# compressible (it is also the file that needs the largest match offsets).
#
# Every container is built to be structurally consistent -- box sizes, chunk
# offsets, Matroska seek positions and AVI indexes all agree with the bytes --
# so a real demuxer such as ffprobe can open them.

def _rand_bytes(rng, n):
    return rng.getrandbits(8 * n).to_bytes(n, "little") if n > 0 else b""


def _frame_sizes(rng, n_frames, key_every, key_bytes, delta_bytes):
    """Per-frame coded sizes: big keyframes, smaller log-normal delta frames."""
    sizes = []
    for i in range(n_frames):
        if i % key_every == 0:
            s = int(rng.gauss(key_bytes, key_bytes * 0.06))
        else:
            s = int(rng.lognormvariate(math.log(delta_bytes), 0.45))
        sizes.append(max(256, s))
    return sizes


# ---- H.264 samples (length-prefixed "AVCC" form, as used in MP4 and MKV) -----

_SPS = bytes.fromhex("6764001facd9405005bb011000000300100000030300f1831960")
_PPS = bytes.fromhex("68ebe3cb22c0")

_X264_TEXT = (
    "x264 - core 164 r3095 baee400 - H.264/MPEG-4 AVC codec - Copyleft 2003-2022 - "
    "http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 "
    "analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 "
    "chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 "
    "chroma_qp_offset=-2 threads=3 lookahead_threads=1 sliced_threads=0 nr=0 "
    "decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 "
    "b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 "
    "keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf "
    "mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00"
)


def _sei_user_data(text):
    uuid = bytes.fromhex("dc45e9bde6d948b7962cd820d923eeef")  # x264's SEI UUID
    body = uuid + text.encode("ascii") + b"\x00"
    size = len(body)
    return b"\x05" + b"\xff" * (size // 255) + bytes([size % 255]) + body + b"\x80"


def _avc_nal(nal_type, ref_idc, body):
    return struct.pack(">I", 1 + len(body)) + bytes([(ref_idc << 5) | nal_type]) + body


def _h264_sample(rng, size, keyframe, frame_no, with_sei):
    parts = []
    used = 0
    if with_sei:
        sei = _avc_nal(6, 0, _sei_user_data(_X264_TEXT))
        parts.append(sei)
        used += len(sei)
    if keyframe:
        slice_hdr = b"\x88\x84\x21"
    else:
        slice_hdr = bytes([0x9A, 0x20 | (frame_no & 0x0F), 0x6C])
    payload_len = max(0, size - used - 5 - len(slice_hdr))
    parts.append(_avc_nal(5 if keyframe else 1, 3 if keyframe else 2,
                          slice_hdr + _rand_bytes(rng, payload_len)))
    return b"".join(parts)


def _avcc_record():
    return (bytes([1, _SPS[1], _SPS[2], _SPS[3], 0xFF, 0xE1])
            + struct.pack(">H", len(_SPS)) + _SPS
            + bytes([1]) + struct.pack(">H", len(_PPS)) + _PPS)


# ---- ISO base media: MP4 / MOV ------------------------------------------------

def _box(typ, payload=b""):
    return struct.pack(">I4s", 8 + len(payload), typ) + payload


def _fullbox(typ, version, flags, payload=b""):
    return _box(typ, bytes([version]) + flags.to_bytes(3, "big") + payload)


_MAC_EPOCH_2024 = 3786912000      # 2024-01-01T00:00:00Z, seconds since 1904
_UNITY_MATRIX = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


def _mp4_moov(width, height, timescale, delta, sizes, key_numbers, offsets, quicktime):
    n = len(sizes)
    movie_ts = 1000
    dur_track = n * delta
    dur_movie = dur_track * movie_ts // timescale
    t = _MAC_EPOCH_2024

    mvhd = _fullbox(b"mvhd", 0, 0,
                    struct.pack(">IIII", t, t, movie_ts, dur_movie)
                    + struct.pack(">I", 0x00010000) + struct.pack(">H", 0x0100)
                    + b"\x00" * 10 + _UNITY_MATRIX + b"\x00" * 24
                    + struct.pack(">I", 2))

    tkhd = _fullbox(b"tkhd", 0, 3,
                    struct.pack(">IIII", t, t, 1, 0) + struct.pack(">I", dur_movie)
                    + b"\x00" * 8 + struct.pack(">hhhh", 0, 0, 0, 0)
                    + _UNITY_MATRIX + struct.pack(">II", width << 16, height << 16))

    mdhd = _fullbox(b"mdhd", 0, 0,
                    struct.pack(">IIII", t, t, timescale, dur_track)
                    + struct.pack(">HH", 0x55C4, 0))          # language 'und'

    if quicktime:
        hdlr = _fullbox(b"hdlr", 0, 0, b"mhlr" + b"vide" + b"appl"
                        + struct.pack(">II", 0, 0) + b"\x0cVideoHandler")
        dhlr = _fullbox(b"hdlr", 0, 0, b"dhlr" + b"alis" + b"appl"
                        + struct.pack(">II", 0, 0) + b"\x0bDataHandler")
    else:
        hdlr = _fullbox(b"hdlr", 0, 0, struct.pack(">I4s", 0, b"vide")
                        + b"\x00" * 12 + b"VideoHandler\x00")
        dhlr = b""

    vmhd = _fullbox(b"vmhd", 0, 1, struct.pack(">HHHH", 0, 0, 0, 0))
    dinf = _box(b"dinf", _fullbox(b"dref", 0, 0,
                                  struct.pack(">I", 1) + _fullbox(b"url ", 0, 1)))

    avc1 = (b"\x00" * 6 + struct.pack(">H", 1)
            + struct.pack(">HH", 0, 0) + b"\x00" * 12
            + struct.pack(">HH", width, height)
            + struct.pack(">II", 0x00480000, 0x00480000)
            + struct.pack(">I", 0) + struct.pack(">H", 1)
            + b"\x0aAVC Coding".ljust(32, b"\x00")
            + struct.pack(">Hh", 0x18, -1)
            + _box(b"avcC", _avcc_record()))
    stsd = _fullbox(b"stsd", 0, 0, struct.pack(">I", 1) + _box(b"avc1", avc1))
    stts = _fullbox(b"stts", 0, 0, struct.pack(">III", 1, n, delta))
    stss = _fullbox(b"stss", 0, 0, struct.pack(">I", len(key_numbers))
                    + b"".join(struct.pack(">I", k) for k in key_numbers))
    stsc = _fullbox(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, 1, 1))
    stsz = _fullbox(b"stsz", 0, 0, struct.pack(">II", 0, n)
                    + b"".join(struct.pack(">I", s) for s in sizes))
    stco = _fullbox(b"stco", 0, 0, struct.pack(">I", n)
                    + b"".join(struct.pack(">I", o) for o in offsets))
    stbl = _box(b"stbl", stsd + stts + stss + stsc + stsz + stco)

    minf = _box(b"minf", vmhd + dhlr + dinf + stbl)
    trak = _box(b"trak", tkhd + _box(b"mdia", mdhd + hdlr + minf))

    udta = b""
    if not quicktime:
        too = _box(b"\xa9too", _box(b"data", struct.pack(">II", 1, 0) + b"Lavf60.16.100"))
        mdir = _fullbox(b"hdlr", 0, 0, struct.pack(">I4s", 0, b"mdir")
                        + b"appl" + b"\x00" * 8 + b"\x00")
        udta = _box(b"udta", _fullbox(b"meta", 0, 0, mdir + _box(b"ilst", too)))

    return _box(b"moov", mvhd + trak + udta)


def gen_mp4(faststart, seed, quicktime=False, n_frames=120, width=640, height=360,
            key_every=30):
    """H.264-style MP4 / MOV. faststart=True puts moov before mdat."""
    rng = random.Random(seed)
    plan = _frame_sizes(rng, n_frames, key_every, 14_000, 2_200)
    samples = [_h264_sample(rng, sz, i % key_every == 0, i, i == 0)
               for i, sz in enumerate(plan)]
    sizes = [len(s) for s in samples]
    keys = [i + 1 for i in range(n_frames) if i % key_every == 0]
    mdat = _box(b"mdat", b"".join(samples))

    if quicktime:
        ftyp = _box(b"ftyp", b"qt  " + struct.pack(">I", 0x200) + b"qt  ")
    else:
        ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 0x200) + b"isomiso2avc1mp41")

    def moov_for(first_sample_offset):
        offsets, o = [], first_sample_offset
        for s in sizes:
            offsets.append(o)
            o += s
        return _mp4_moov(width, height, 15360, 512, sizes, keys, offsets, quicktime)

    if faststart:
        moov_len = len(moov_for(0))               # length does not depend on offsets
        return ftyp + moov_for(len(ftyp) + moov_len + 8) + mdat
    pad = _box(b"wide") if quicktime else _box(b"free")
    return ftyp + pad + mdat + moov_for(len(ftyp) + len(pad) + 8)


# ---- Matroska / WebM ----------------------------------------------------------

def _ebml_vint(n):
    for w in range(1, 9):
        if n < (1 << (7 * w)) - 1:
            return (n | (1 << (7 * w))).to_bytes(w, "big")
    raise ValueError("EBML size too large")


def _ebml_size8(n):
    return (n | (1 << 56)).to_bytes(8, "big")


def _el(eid, payload):
    return eid + _ebml_vint(len(payload)) + payload


def _uint_be(n, width=None):
    if width is None:
        width = max(1, (n.bit_length() + 7) // 8)
    return n.to_bytes(width, "big")


def _ebml_void(total):
    """A Void element occupying exactly `total` bytes (total >= 129)."""
    body = total - 1 - len(_ebml_vint(total - 3))
    return b"\xEC" + _ebml_vint(body) + b"\x00" * body


_ID_INFO, _ID_TRACKS, _ID_CUES = b"\x15\x49\xA9\x66", b"\x16\x54\xAE\x6B", b"\x1C\x53\xBB\x6B"


def gen_matroska(webm, seed, n_frames=120, width=640, height=360, key_every=30):
    """VP9-style WebM or H.264-style MKV with Cues and SeekHead consistent."""
    rng = random.Random(seed)
    key_bytes, delta_bytes = (12_000, 1_900) if webm else (14_000, 2_200)
    plan = _frame_sizes(rng, n_frames, key_every, key_bytes, delta_bytes)

    frames = []
    for i, sz in enumerate(plan):
        key = i % key_every == 0
        if webm:
            hdr = bytes([0x82 if key else 0x86, 0x49, 0x83 if key else 0x81])
            data = hdr + _rand_bytes(rng, max(0, sz - 3))
        else:
            data = _h264_sample(rng, sz, key, i, i == 0)
        frames.append((key, data))

    ms = lambda i: round(i * 1000 / 30)
    clusters = []                                   # (cluster_time_ms, element_bytes)
    for c0 in range(0, n_frames, key_every):
        tc = ms(c0)
        body = _el(b"\xE7", _uint_be(tc))
        for i in range(c0, min(c0 + key_every, n_frames)):
            key, data = frames[i]
            block = (b"\x81" + struct.pack(">h", ms(i) - tc)
                     + bytes([0x80 if key else 0x00]) + data)
            body += _el(b"\xA3", block)
        clusters.append((tc, _el(b"\x1F\x43\xB6\x75", body)))

    ebml = _el(b"\x1A\x45\xDF\xA3",
               _el(b"\x42\x86", _uint_be(1)) + _el(b"\x42\xF7", _uint_be(1))
               + _el(b"\x42\xF2", _uint_be(4)) + _el(b"\x42\xF3", _uint_be(8))
               + _el(b"\x42\x82", b"webm" if webm else b"matroska")
               + _el(b"\x42\x87", _uint_be(4)) + _el(b"\x42\x85", _uint_be(2)))

    info = _el(_ID_INFO,
               _el(b"\x2A\xD7\xB1", _uint_be(1_000_000))
               + _el(b"\x4D\x80", b"Lavf60.16.100") + _el(b"\x57\x41", b"Lavf60.16.100")
               + _el(b"\x44\x89", struct.pack(">d", float(ms(n_frames))))
               + _el(b"\x73\xA4", _rand_bytes(rng, 16)))

    video = _el(b"\xE0", _el(b"\xB0", _uint_be(width)) + _el(b"\xBA", _uint_be(height)))
    entry = (_el(b"\xD7", _uint_be(1)) + _el(b"\x73\xC5", _rand_bytes(rng, 8))
             + _el(b"\x83", _uint_be(1)) + _el(b"\x9C", _uint_be(0))
             + _el(b"\x23\xE3\x83", _uint_be(33_333_333))
             + _el(b"\x86", b"V_VP9" if webm else b"V_MPEG4/ISO/AVC")
             + (b"" if webm else _el(b"\x63\xA2", _avcc_record()))
             + video)
    tracks = _el(_ID_TRACKS, _el(b"\xAE", entry))

    def seek(sid, pos):
        return _el(b"\x4D\xBB", _el(b"\x53\xAB", sid) + _el(b"\x53\xAC", _uint_be(pos, 4)))

    def seekhead(p_info, p_tracks, p_cues):
        return _el(b"\x11\x4D\x9B\x74",
                   seek(_ID_INFO, p_info) + seek(_ID_TRACKS, p_tracks) + seek(_ID_CUES, p_cues))

    void = _ebml_void(1024 if webm else 4096)
    p_info = len(seekhead(0, 0, 0)) + len(void)
    p_tracks = p_info + len(info)
    pos = p_tracks + len(tracks)
    cue_points = []
    for tc, el in clusters:
        cue_points.append(_el(b"\xBB", _el(b"\xB3", _uint_be(tc))
                              + _el(b"\xB7", _el(b"\xF7", _uint_be(1))
                                    + _el(b"\xF1", _uint_be(pos)))))
        pos += len(el)
    cues = _el(_ID_CUES, b"".join(cue_points))
    payload = (seekhead(p_info, p_tracks, pos) + void + info + tracks
               + b"".join(el for _, el in clusters) + cues)
    return ebml + b"\x18\x53\x80\x67" + _ebml_size8(len(payload)) + payload


# ---- AVI (MJPEG) --------------------------------------------------------------

_STD_LUMA_Q = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]
_STD_CHROMA_Q = (
    [17, 18, 24, 47] + [99] * 4 + [18, 21, 26, 66] + [99] * 4
    + [24, 26, 56] + [99] * 5 + [47, 66] + [99] * 6 + [99] * 32
)


def _jpeg_headers(width, height):
    q = lambda tbl: bytes(max(1, (v * 30 + 50) // 100) for v in tbl)   # ~quality 85
    dht = lambda cls_id, counts, symbols: (
        b"\xFF\xC4" + struct.pack(">H", 2 + 1 + 16 + len(symbols))
        + bytes([cls_id]) + bytes(counts) + bytes(symbols))
    ac_symbols = [(r << 4) | s for r in range(16) for s in range(1, 11)] + [0x00, 0xF0]
    random.Random(1234).shuffle(ac_symbols)
    return (b"\xFF\xD8"
            + b"\xFF\xE0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00"
            + struct.pack(">HH", 1, 1) + b"\x00\x00"
            + b"\xFF\xDB" + struct.pack(">H", 67) + b"\x00" + q(_STD_LUMA_Q)
            + b"\xFF\xDB" + struct.pack(">H", 67) + b"\x01" + q(_STD_CHROMA_Q)
            + b"\xFF\xC0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", height, width)
            + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
            + dht(0x00, [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], range(12))
            + dht(0x10, [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D], ac_symbols)
            + dht(0x01, [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], range(12))
            + dht(0x11, [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77], ac_symbols)
            + b"\xFF\xDA" + struct.pack(">H", 12) + b"\x03\x01\x00\x02\x11\x03\x11\x00\x3F\x00")


def _riff_chunk(fourcc, data):
    return fourcc + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) % 2 else b"")


def _riff_list(kind, children):
    return b"LIST" + struct.pack("<I", 4 + len(children)) + kind + children


def gen_mjpeg_avi(seed, n_frames=20, width=320, height=240):
    rng = random.Random(seed)
    headers = _jpeg_headers(width, height)            # identical in every frame
    chunks, entries, offset = [], [], 4               # offsets count from the 'movi' fourcc
    for i in range(n_frames):
        scan = _rand_bytes(rng, max(1024, int(rng.gauss(18_000, 1_500)))).replace(b"\xff", b"\xff\x00")
        frame = headers + scan + b"\xFF\xD9"
        chunk = _riff_chunk(b"00dc", frame)
        entries.append(struct.pack("<4sIII", b"00dc", 0x10, offset, len(frame)))
        chunks.append(chunk)
        offset += len(chunk)
    max_frame = max(len(c) for c in chunks)

    avih = struct.pack("<IIIIIIIIII4I", 33333, max_frame * 30, 0, 0x910, n_frames, 0, 1,
                       max_frame, width, height, 0, 0, 0, 0)
    strh = (b"vids" + b"MJPG" + struct.pack("<IHHIIIIIIIIhhhh", 0, 0, 0, 0, 1, 30, 0, n_frames,
                                             max_frame, 0xFFFFFFFF, 0, 0, 0, width, height))
    strf = struct.pack("<IiiHH4sIiiII", 40, width, height, 1, 24, b"MJPG",
                       width * height * 3, 0, 0, 0, 0)
    hdrl = _riff_list(b"hdrl", _riff_chunk(b"avih", avih)
                      + _riff_list(b"strl", _riff_chunk(b"strh", strh) + _riff_chunk(b"strf", strf)))
    junk = _riff_chunk(b"JUNK", b"\x00" * 1008)
    info = _riff_list(b"INFO", _riff_chunk(b"ISFT", b"Lavf60.16.100\x00"))
    movi = _riff_list(b"movi", b"".join(chunks))
    idx1 = _riff_chunk(b"idx1", b"".join(entries))
    body = b"AVI " + hdrl + junk + info + movi + idx1
    return b"RIFF" + struct.pack("<I", len(body)) + body


# ---- MPEG transport stream ----------------------------------------------------

def _crc32_mpeg(data):
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
    return crc


def _ts_psi(pid, section, cc):
    section += struct.pack(">I", _crc32_mpeg(section))
    payload = (b"\x00" + section).ljust(184, b"\xFF")
    return bytes([0x47, 0x40 | (pid >> 8), pid & 0xFF, 0x10 | (cc & 0x0F)]) + payload


def _pes_header(pts):
    p = bytes([0x21 | ((pts >> 29) & 0x0E), (pts >> 22) & 0xFF,
               ((pts >> 14) & 0xFE) | 1, (pts >> 7) & 0xFF, ((pts << 1) & 0xFE) | 1])
    return b"\x00\x00\x01\xE0" + struct.pack(">H", 0) + b"\x80\x80\x05" + p


def _ts_packets(pes, pid, cc, pcr=None, keyframe=False):
    packets, pos, first = [], 0, True
    while pos < len(pes):
        remaining = len(pes) - pos
        af = b""
        if first and pcr is not None:
            pcr_field = struct.pack(">Q", (pcr << 15) | (0x3F << 9))[2:]
            af = bytes([7, 0x10 | (0x40 if keyframe else 0)]) + pcr_field
        space = 184 - len(af)
        if remaining < space:                         # stuff with an adaptation field
            stuff = space - remaining
            if af:
                af = bytes([af[0] + stuff]) + af[1:] + b"\xFF" * stuff
            elif stuff == 1:
                af = b"\x00"
            else:
                af = bytes([stuff - 1, 0x00]) + b"\xFF" * (stuff - 2)
            space = 184 - len(af)
        chunk = pes[pos:pos + space]
        pos += len(chunk)
        hdr = bytes([0x47, (0x40 if first else 0) | (pid >> 8), pid & 0xFF,
                     (0x30 if af else 0x10) | (cc & 0x0F)])
        packets.append(hdr + af + chunk)
        cc = (cc + 1) & 0x0F
        first = False
    return packets, cc


def gen_mpegts(seed, n_frames=100, key_every=25):
    rng = random.Random(seed)
    plan = _frame_sizes(rng, n_frames, key_every, 14_000, 2_800)
    pat = b"\x00" + struct.pack(">H", 0xB000 | 13) + struct.pack(">H", 1) + b"\xC1\x00\x00" \
          + struct.pack(">HH", 1, 0xE000 | 0x1000)
    pmt = b"\x02" + struct.pack(">H", 0xB000 | 18) + struct.pack(">H", 1) + b"\xC1\x00\x00" \
          + struct.pack(">H", 0xE000 | 0x100) + struct.pack(">H", 0xF000) \
          + bytes([0x1B]) + struct.pack(">HH", 0xE000 | 0x100, 0xF000)
    sc = b"\x00\x00\x00\x01"
    out, cc_vid, cc_pat, cc_pmt = [], 0, 0, 0
    for i, size in enumerate(plan):
        key = i % key_every == 0
        if i % 5 == 0:                                # PSI repeated every 5 frames
            out.append(_ts_psi(0, pat, cc_pat)); cc_pat = (cc_pat + 1) & 0x0F
            out.append(_ts_psi(0x1000, pmt, cc_pmt)); cc_pmt = (cc_pmt + 1) & 0x0F
        es = sc + b"\x09\xF0"                         # access unit delimiter
        if key:
            es += sc + _SPS + sc + _PPS
        hdr = b"\x65\x88\x84\x21" if key else bytes([0x41, 0x9A, 0x20 | (i & 0x0F), 0x6C])
        es += sc + hdr + _rand_bytes(rng, max(0, size - len(es) - len(sc) - len(hdr)))
        pts = 90000 + i * 3000
        pkts, cc_vid = _ts_packets(_pes_header(pts) + es, 0x100, cc_vid,
                                   pcr=(pts - 6000) if key else None, keyframe=key)
        out.extend(pkts)
    return b"".join(out)


# ---- Raw video: YUV4MPEG2 (genuinely compressible) ------------------------------

def gen_y4m(width=176, height=144, n_frames=12, seed=11):
    rng = random.Random(seed)
    out = bytearray(f"YUV4MPEG2 W{width} H{height} F30:1 Ip A1:1 C420jpeg XYSCSS=420JPEG\n".encode())
    cw, ch = width // 2, height // 2
    noise = (-1, 0, 0, 1)
    for f in range(n_frames):
        out += b"FRAME\n"
        bx = int((width - 24) * (0.5 + 0.5 * math.sin(f * 0.45)))
        by = int((height - 24) * (0.5 + 0.5 * math.cos(f * 0.31)))
        for y in range(height):                       # luma: gradient + checks + moving box + sensor noise
            base = 40 + (y * 150) // height
            row = bytearray()
            for x in range(width):
                v = base + (x * 30) // width + 8 * (((x >> 4) + (y >> 4)) & 1)
                if bx <= x < bx + 24 and by <= y < by + 24:
                    v = 235
                v += noise[rng.getrandbits(2)]
                row.append(min(235, max(16, v)))
            out += row
        for plane in (0, 1):                          # chroma: smooth gradients, box tinted
            for y in range(ch):
                row = bytearray()
                for x in range(cw):
                    v = 128 + (((x * 40) // cw) if plane == 0 else ((y * 40) // ch)) - 20
                    if bx // 2 <= x < bx // 2 + 12 and by // 2 <= y < by // 2 + 12:
                        v = 90 if plane == 0 else 170
                    row.append(v)
                out += row
    return bytes(out)


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

    print("\n── Video containers ─────────────────────────────────────────────")
    out("clip_h264_faststart.mp4", gen_mp4(faststart=True, seed=21),
        "MP4 (H.264-style) moov-first (faststart) — structured moov up front, random coded frames")
    out("clip_h264.mp4",           gen_mp4(faststart=False, seed=22),
        "MP4 (H.264-style) moov-last (ffmpeg default layout) — mdat first, moov at the end")
    out("clip.mov",                gen_mp4(faststart=False, quicktime=True, seed=23),
        "QuickTime MOV — wide + mdat, then moov (qt brand, mhlr/dhlr handlers)")
    out("clip_vp9.webm",           gen_matroska(webm=True, seed=24),
        "WebM (VP9-style) — EBML header, SeekHead, Cues, 1 s clusters, random coded frames")
    out("clip_h264.mkv",           gen_matroska(webm=False, seed=25),
        "Matroska (H.264-style) — Void padding, avcC CodecPrivate, Cues")
    out("clip_mjpeg.avi",          gen_mjpeg_avi(seed=26),
        "AVI MJPEG 320×240 — identical JPEG tables repeated in every frame, idx1 index")
    out("clip.ts",                 gen_mpegts(seed=27),
        "MPEG-TS H.264 — 188-byte packets, PAT/PMT repeated, continuity counters, PCR")
    out("clip_raw.y4m",            gen_y4m(),
        "Raw YUV4MPEG2 4:2:0 176×144 ×12 frames — uncompressed (the compressible video case)")

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
