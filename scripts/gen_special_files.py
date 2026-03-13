#!/usr/bin/env python3
# scripts/gen_special_files.py
# Generates all special benchmark files and writes special_manifest.json
# Usage: python3 scripts/gen_special_files.py [--output-dir bench_files_special]

import argparse
import json
import math
import os
import random
import struct
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="bench_files_special")
    return p.parse_args()

def write(path, data, mode="wb"):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, mode) as f:
        if isinstance(data, str):
            f.write(data.encode())
        else:
            f.write(data)
    return os.path.getsize(path)

# ── STL binary ────────────────────────────────────────────────────────────────
def gen_stl_binary(path):
    random.seed(42)
    n_tris = 5000
    header = b"MBFA benchmark STL binary mesh" + b"\x00" * (80 - 30)
    data = header + struct.pack("<I", n_tris)
    for i in range(n_tris):
        phi   = math.pi * i / n_tris
        theta = 2 * math.pi * i / 100
        nx = math.sin(phi) * math.cos(theta)
        ny = math.cos(phi)
        nz = math.sin(phi) * math.sin(theta)
        data += struct.pack("<fff", nx, ny, nz)
        for _ in range(3):
            vx = nx * 10.0 + random.gauss(0, 0.01)
            vy = ny * 10.0 + random.gauss(0, 0.01)
            vz = nz * 10.0 + random.gauss(0, 0.01)
            data += struct.pack("<fff", vx, vy, vz)
        data += struct.pack("<H", 0)
    return write(path, data)

# ── PLY binary ────────────────────────────────────────────────────────────────
def gen_ply_binary(path):
    random.seed(7)
    n_verts = 2000
    n_faces = 1800
    hdr  = "ply\n"
    hdr += "format binary_little_endian 1.0\n"
    hdr += f"element vertex {n_verts}\n"
    hdr += "property float x\nproperty float y\nproperty float z\n"
    hdr += "property float nx\nproperty float ny\nproperty float nz\n"
    hdr += "property float s\nproperty float t\n"
    hdr += f"element face {n_faces}\n"
    hdr += "property list uchar int vertex_indices\n"
    hdr += "end_header\n"
    body = hdr.encode()
    for i in range(n_verts):
        phi   = math.pi * i / n_verts
        theta = 2 * math.pi * (i % 100) / 100
        x = math.sin(phi) * math.cos(theta) * 5.0 + random.gauss(0, 0.005)
        y = math.cos(phi) * 5.0
        z = math.sin(phi) * math.sin(theta) * 5.0
        nx, ny, nz = math.sin(phi)*math.cos(theta), math.cos(phi), math.sin(phi)*math.sin(theta)
        s, t = theta / (2*math.pi), phi / math.pi
        body += struct.pack("<ffffffff", x, y, z, nx, ny, nz, s, t)
    for i in range(n_faces):
        a = i % n_verts
        b = (i + 1) % n_verts
        c = (i + 7) % n_verts
        body += struct.pack("<Biii", 3, a, b, c)
    return write(path, body)

# ── GLB stub ──────────────────────────────────────────────────────────────────
def gen_glb(path):
    random.seed(13)
    json_str = json.dumps({
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 500, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 1200, "type": "SCALAR"}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 6000},
            {"buffer": 0, "byteOffset": 6000, "byteLength": 2400}
        ],
        "buffers": [{"byteLength": 8400}]
    })
    json_bytes = json_str.encode()
    pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * pad
    bin_data = b""
    for i in range(500):
        phi   = math.pi * i / 500
        theta = 2 * math.pi * (i % 50) / 50
        x = math.sin(phi)*math.cos(theta)*3 + random.gauss(0, 0.005)
        y = math.cos(phi)*3
        z = math.sin(phi)*math.sin(theta)*3
        bin_data += struct.pack("<fff", x, y, z)
    for i in range(1200):
        bin_data += struct.pack("<H", i % 500)
    bin_pad = (4 - len(bin_data) % 4) % 4
    bin_data += b"\x00" * bin_pad
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk  = struct.pack("<II", len(bin_data),  0x004E4942) + bin_data
    body       = json_chunk + bin_chunk
    total_len  = 12 + len(body)
    glb = struct.pack("<III", 0x46546C67, 2, total_len) + body
    return write(path, glb)

# ── Unity C# scripts ──────────────────────────────────────────────────────────
def gen_unity_csharp(path):
    lines = []
    lines.append("using UnityEngine;")
    lines.append("using UnityEngine.UI;")
    lines.append("using System.Collections;")
    lines.append("using System.Collections.Generic;")
    lines.append("")
    components = ["PlayerController","EnemyAI","GameManager","UIManager","AudioManager",
                  "InputHandler","CameraController","InventorySystem","QuestTracker","SaveSystem"]
    for comp in components * 8:
        lines.append(f"public class {comp} : MonoBehaviour")
        lines.append("{")
        lines.append(f"    [SerializeField] private float speed = 5f;")
        lines.append(f"    [SerializeField] private int health = 100;")
        lines.append(f"    private Rigidbody _rb;")
        lines.append(f"    private bool _isActive = true;")
        lines.append("")
        lines.append(f"    void Awake() {{ _rb = GetComponent<Rigidbody>(); }}")
        lines.append(f"    void Start() {{ Debug.Log(\"{comp} started\"); }}")
        lines.append(f"    void Update()")
        lines.append(f"    {{")
        lines.append(f"        if (!_isActive) return;")
        lines.append(f"        float h = Input.GetAxis(\"Horizontal\");")
        lines.append(f"        float v = Input.GetAxis(\"Vertical\");")
        lines.append(f"        _rb.MovePosition(transform.position + new Vector3(h, 0, v) * speed * Time.deltaTime);")
        lines.append(f"    }}")
        lines.append(f"    public void TakeDamage(int amount) {{ health -= amount; if (health <= 0) Die(); }}")
        lines.append(f"    private void Die() {{ _isActive = false; gameObject.SetActive(false); }}")
        lines.append("}")
        lines.append("")
    return write(path, "\n".join(lines))

# ── Unity YAML asset ──────────────────────────────────────────────────────────
def gen_unity_yaml(path):
    random.seed(3)
    lines = ["%YAML 1.1", "%TAG !u! tag:unity3d.com,2011:",""]
    for i in range(200):
        lines.append(f"--- !u!1 &{random.randint(100000000,999999999)}")
        lines.append("GameObject:")
        lines.append(f"  m_ObjectHideFlags: 0")
        lines.append(f"  m_CorrespondingSourceObject: {{fileID: 0}}")
        lines.append(f"  m_PrefabInstance: {{fileID: 0}}")
        lines.append(f"  m_PrefabAsset: {{fileID: 0}}")
        lines.append(f"  serializedVersion: 6")
        lines.append(f"  m_Component:")
        lines.append(f"  - component: {{fileID: {random.randint(100000000,999999999)}}}")
        lines.append(f"  m_Layer: 0")
        lines.append(f"  m_Name: GameObject_{i}")
        lines.append(f"  m_TagString: Untagged")
        lines.append(f"  m_IsActive: 1")
        lines.append(f"--- !u!4 &{random.randint(100000000,999999999)}")
        lines.append("Transform:")
        lines.append(f"  m_LocalPosition: {{x: {random.uniform(-100,100):.5f}, y: {random.uniform(-10,100):.5f}, z: {random.uniform(-100,100):.5f}}}")
        lines.append(f"  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}")
        lines.append(f"  m_LocalScale: {{x: 1, y: 1, z: 1}}")
        lines.append(f"  m_Children: []")
        lines.append(f"  m_Father: {{fileID: 0}}")
        lines.append("")
    return write(path, "\n".join(lines))

# ── Unity terrain .raw ────────────────────────────────────────────────────────
def gen_terrain_raw(path, w=513, h=513, seed_x=0.05, seed_y=0.05):
    data = b""
    for y in range(h):
        for x in range(w):
            height = int(
                (math.sin(x * seed_x) * math.cos(y * seed_y) * 0.4 +
                 math.sin(x * 0.02 + y * 0.03) * 0.3 +
                 math.sin(x * 0.1) * 0.15 + 0.5) * 65535
            )
            height = max(0, min(65535, height))
            data += struct.pack("<H", height)
    return write(path, data)

# ── Unity shader ──────────────────────────────────────────────────────────────
def gen_unity_shader(path):
    lines = []
    shaders = ["StandardLit","Unlit","Toon","WaterSurface","ParticleAdditive","SkyboxProc"]
    for name in shaders * 5:
        lines.append(f'Shader "Custom/{name}"')
        lines.append("{")
        lines.append("    Properties")
        lines.append("    {")
        lines.append('        _MainTex ("Texture", 2D) = "white" {}')
        lines.append('        _Color ("Color", Color) = (1,1,1,1)')
        lines.append('        _Glossiness ("Smoothness", Range(0,1)) = 0.5')
        lines.append('        _Metallic ("Metallic", Range(0,1)) = 0.0')
        lines.append("    }")
        lines.append("    SubShader")
        lines.append("    {")
        lines.append('        Tags { "RenderType"="Opaque" }')
        lines.append("        CGPROGRAM")
        lines.append('        #pragma surface surf Standard fullforwardshadows')
        lines.append('        #pragma target 3.0')
        lines.append("        sampler2D _MainTex;")
        lines.append("        struct Input { float2 uv_MainTex; };")
        lines.append("        half _Glossiness; half _Metallic; fixed4 _Color;")
        lines.append("        void surf (Input IN, inout SurfaceOutputStandard o) {")
        lines.append("            fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;")
        lines.append("            o.Albedo = c.rgb; o.Metallic = _Metallic;")
        lines.append("            o.Smoothness = _Glossiness; o.Alpha = c.a;")
        lines.append("        }")
        lines.append("        ENDCG")
        lines.append("    }")
        lines.append("}")
        lines.append("")
    return write(path, "\n".join(lines))

# ── Unreal ini config ─────────────────────────────────────────────────────────
def gen_unreal_ini(path):
    sections = ["Engine.RendererSettings","Engine.PhysicsSettings","Engine.GameViewportClient",
                "/Script/Engine.GameModeBase","/Script/Engine.PlayerInput",
                "/Script/UnrealEd.EditorEngine"]
    lines = []
    for sec in sections * 15:
        lines.append(f"[{sec}]")
        lines.append("bEnableRayTracing=False")
        lines.append("r.Shadow.CSM.MaxCascades=4")
        lines.append("r.DefaultFeature.AmbientOcclusion=True")
        lines.append("r.DefaultFeature.Bloom=True")
        lines.append("bSubsteping=False")
        lines.append("MaxPhysicsDeltaTime=0.033333")
        lines.append("bSupportUTF8Ini=True")
        lines.append("DefaultBuildSettings=BuildSettingsVersion_V4")
        lines.append("+ActiveGameNameRedirects=(OldGameName=\"/Script/Engine\",NewGameName=\"/Script/MyGame\")")
        lines.append("")
    return write(path, "\n".join(lines))

# ── Unreal uasset stub ────────────────────────────────────────────────────────
def gen_uasset_stub(path):
    random.seed(99)
    magic   = struct.pack("<I", 0x9E2A83C1)
    version = struct.pack("<ii", -8, 0)
    props = b""
    prop_names = [b"StaticMesh\x00", b"Material\x00\x00\x00", b"Transform\x00\x00",
                  b"Mobility\x00\x00\x00", b"CastShadow\x00\x00"]
    for i in range(300):
        name   = prop_names[i % len(prop_names)]
        floats = struct.pack("<fff",
            math.sin(i * 0.1) * 100,
            math.cos(i * 0.1) * 100,
            i * 0.5)
        props += name + floats + struct.pack("<I", i)
    data = magic + version + props + b"\x00" * 16
    return write(path, data)

# ── Unreal uplugin JSON ───────────────────────────────────────────────────────
def gen_uplugin(path):
    plugins = []
    categories = ["Rendering", "Physics", "AI", "Network", "UI"]
    platform_list = ["Win64", "Mac", "Linux", "Android", "IOS"]
    for i in range(50):
        plugins.append({
            "Name": f"Plugin{i:03d}",
            "Enabled": (i % 3 != 0),
            "MarketplaceURL": f"https://marketplace.unrealengine.com/product/plugin{i:03d}",
            "SupportedTargetPlatforms": platform_list,
            "VersionName": f"1.{i}.0",
            "FriendlyName": f"Awesome Plugin {i:03d}",
            "Description": f"This plugin provides functionality number {i} for Unreal Engine projects.",
            "Category": categories[i % 5],
            "bIsBetaVersion": (i % 7 == 0),
            "Modules": [{"Name": f"Plugin{i:03d}Module", "Type": "Runtime", "LoadingPhase": "Default"}]
        })
    obj = {"FileVersion": 3, "Version": 1, "VersionName": "1.0.0",
           "FriendlyName": "MBFA Test Plugin Bundle", "Plugins": plugins}
    return write(path, json.dumps(obj, indent=2).encode())

# ── Generic YAML config ───────────────────────────────────────────────────────
def gen_yaml_config(path):
    random.seed(11)
    lines = []
    services = ["api","worker","scheduler","cache","db","proxy","monitor","logger"]
    envs = ["development","staging","production"]
    region_list = ["us-east-1", "eu-west-1", "ap-southeast-1"]
    for env in envs:
        for svc in services * 4:
            lines.append(f"# {env} {svc} configuration")
            lines.append(f"{svc}_{env}:")
            lines.append(f"  image: myregistry.io/{svc}:latest")
            lines.append(f"  replicas: {random.randint(1,5)}")
            lines.append(f"  resources:")
            lines.append(f"    requests:")
            lines.append(f'      cpu: "100m"')
            lines.append(f'      memory: "128Mi"')
            lines.append(f"    limits:")
            lines.append(f'      cpu: "500m"')
            lines.append(f'      memory: "512Mi"')
            lines.append(f"  env:")
            lines.append(f"    - name: APP_ENV")
            lines.append(f"      value: {env}")
            lines.append(f"    - name: LOG_LEVEL")
            lines.append(f"      value: info")
            lines.append(f"  ports:")
            lines.append(f"    - containerPort: 8080")
            lines.append(f"  livenessProbe:")
            lines.append(f"    httpGet:")
            lines.append(f"      path: /health")
            lines.append(f"      port: 8080")
            lines.append(f"    initialDelaySeconds: 30")
            lines.append(f"    periodSeconds: 10")
            lines.append("")
    return write(path, "\n".join(lines).encode())

# ── TOML config ───────────────────────────────────────────────────────────────
def gen_toml_config(path):
    region_list = ["us-east-1", "eu-west-1", "ap-southeast-1"]
    lines = []
    for i in range(80):
        region = region_list[i % 3]
        lines.append(f"[[services]]")
        lines.append(f'name = "service_{i:03d}"')
        lines.append(f'host = "192.168.1.{i % 256}"')
        lines.append(f"port = {8000 + i}")
        lines.append(f"timeout = {30 + i % 60}")
        lines.append(f"retries = {3 + i % 5}")
        lines.append(f"enabled = {'true' if i % 3 != 0 else 'false'}")
        lines.append(f'region = "{region}"')
        lines.append(f"[services.tls]")
        lines.append(f"enabled = true")
        lines.append(f'cert = "/etc/ssl/certs/service_{i:03d}.crt"')
        lines.append(f'key = "/etc/ssl/private/service_{i:03d}.key"')
        lines.append(f"[services.metrics]")
        lines.append(f"scrape_interval = 15")
        lines.append(f'endpoint = "/metrics"')
        lines.append("")
    return write(path, "\n".join(lines).encode())

# ── MidMans DixScript .mdix ───────────────────────────────────────────────────
def gen_mdix(path):
    content = r"""
@CONFIG(
    version -> "1.0.0"
    encoding -> "utf-8"
    author -> "MidManStudio"
    created -> 2026-03-12
    features -> "advanced"
    debug_mode -> "off"
    error_handling -> "recover"
    compatibility_mode -> "best_effort"
)

@IMPORTS(
    MathLib from "stdlib/math.mdix" verify "sha256:abc123"
    CryptoLib from_cloud "cloud://libs/crypto" verify "sha256:def456"
    DataUtils from "stdlib/data.mdix"
)

@DLM(
    DCompressor.gzip
    DAuditor.enhanced
    DEncryptor.aes256
)

@ENUMS(
    Status {
        Pending = 0
        Active = 1
        Paused = 2
        Completed = 3
        Failed = 4
    }

    Priority {
        Low = 0
        Medium = 1
        High = 2
        Critical = 3
    }

    Region {
        NorthAmerica = 0
        Europe = 1
        AsiaPacific = 2
        SouthAmerica = 3
    }
)

@QUICKFUNCS(
    ~formatCurrency<string> => global
    (amount<float> currency<string> = "USD") {
        let formatted<string> = $"${amount}";
        if: currency == "EUR" {
            formatted = $"{amount}";
        } elif: currency == "GBP" {
            formatted = $"{amount}";
        }
        return formatted;
    }

    ~clampValue<float> => global
    (value<float> min<float> max<float>) {
        if: value < min { return min; }
        if: value > max { return max; }
        return value;
    }

    ~getStatusLabel<string> => global
    (status<enum>) {
        chk: status {
            -> Status.Pending   { return "Pending Review"; }
            -> Status.Active    { return "Currently Active"; }
            -> Status.Paused    { return "On Hold"; }
            -> Status.Completed { return "Done"; }
            -> miss             { return "Unknown"; }
        }
    }
)

@DATA(
    appName<string> = "MidMans Platform"
    version<string> = "2.1.0"
    buildNumber<int> = 4217
    releaseDate<date> = 2026-03-12
    isProduction<bool> = true
    maxConnections<int> = 1000
    timeout<float> = 30.5f

    server.host = "api.midmans.io"
    server.port = 8443
    server.protocol = "https"
    server.keepAlive = true
    server.maxRetries = 3

    database.host = "db.midmans.io"
    database.port = 5432
    database.name = "midmans_prod"
    database.poolSize = 20
    database.timeout = 5000

    cache.provider = "redis"
    cache.host = "cache.midmans.io"
    cache.port = 6379
    cache.ttl = 3600
    cache.maxMemory = "512mb"

    users :: { id = 1, name = "Alice", status = Status.Active, region = Region.NorthAmerica, balance = 1500.00 }
            { id = 2, name = "Bob", status = Status.Pending, region = Region.Europe, balance = 750.50 }
            { id = 3, name = "Charlie", status = Status.Active, region = Region.AsiaPacific, balance = 2200.75 }
            { id = 4, name = "Diana", status = Status.Completed, region = Region.NorthAmerica, balance = 0.00 }
            { id = 5, name = "Eve", status = Status.Active, region = Region.Europe, balance = 4100.00 }

    features :: "authentication" "authorisation" "billing" "analytics"
               "notifications" "reporting" "export" "import"
               "webhooks" "api_keys" "rate_limiting" "audit_log"
)

@SECURITY(
    encryption -> {
        algorithm = "aes256"
        keySize = 256
        iv = auto
        mode = "gcm"
    }
    validation -> {
        strictMode = true
        maxInputSize = 1048576
        sanitizeHtml = true
    }
    keystore -> {
        provider = "vault"
        endpoint = "https://vault.midmans.io"
        autoRotate = true
        rotationDays = 90
    }
)
"""
    full = (content.strip() + "\n\n") * 12
    return write(path, full.encode())

# ── DLL stub ──────────────────────────────────────────────────────────────────
def gen_dll_stub(path):
    random.seed(55)
    dos = b"MZ" + b"\x00" * 58 + struct.pack("<I", 0x80)
    dos += b"\x00" * (0x80 - len(dos))
    pe_sig = b"PE\x00\x00"
    coff   = struct.pack("<HHIIIHH", 0x14c, 3, 0, 0, 0, 0xe0, 0x0102)
    opt    = struct.pack("<HBBiiiIIIIII", 0x10b, 0, 0, 4096, 512, 512, 0, 0x1000, 0x400000, 8, 8, 4)
    opt   += b"\x00" * (0xe0 - 28)
    pe     = pe_sig + coff + opt
    header = dos + pe + b"\x00" * (0x400 - len(dos + pe))
    opcodes = bytes([0x02, 0x03, 0x04, 0x17, 0x1a, 0x20, 0x28, 0x2a, 0x6f, 0x7b, 0x7c, 0x7d, 0x00])
    body = b""
    for i in range(3000):
        token = struct.pack("<I", 0x0A000001 + (i % 500))
        body += bytes([opcodes[i % len(opcodes)]]) + token
        if i % 20 == 0:
            s = f"MethodName_{i % 100}\x00".encode()
            body += s
    return write(path, header + body)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out  = args.output_dir
    os.makedirs(out, exist_ok=True)

    manifest = []

    def add(name, category, ext, size):
        manifest.append({"name": name, "category": category, "ext": ext,
                         "file": f"{name}{ext}", "bytes": size})

    print(f"Generating special benchmark files -> {out}/")

    size = gen_stl_binary(f"{out}/mesh_sphere.stl")
    add("mesh_sphere", "3D_filter_target", ".stl", size)
    print(f"  mesh_sphere.stl          {size:>10,} bytes")

    size = gen_ply_binary(f"{out}/mesh_sphere.ply")
    add("mesh_sphere_ply", "3D_filter_target", ".ply", size)
    print(f"  mesh_sphere.ply          {size:>10,} bytes")

    size = gen_glb(f"{out}/scene.glb")
    add("scene", "3D_filter_target", ".glb", size)
    print(f"  scene.glb                {size:>10,} bytes")

    size = gen_uasset_stub(f"{out}/StaticMesh.uasset")
    add("StaticMesh", "Unreal", ".uasset", size)
    print(f"  StaticMesh.uasset        {size:>10,} bytes")

    size = gen_uplugin(f"{out}/plugins.json")
    add("plugins", "Unreal", ".json", size)
    print(f"  plugins.json             {size:>10,} bytes")

    size = gen_unreal_ini(f"{out}/DefaultEngine.ini")
    add("DefaultEngine", "Unreal", ".ini", size)
    print(f"  DefaultEngine.ini        {size:>10,} bytes")

    size = gen_unity_csharp(f"{out}/Scripts.cs")
    add("Scripts", "Unity", ".cs", size)
    print(f"  Scripts.cs               {size:>10,} bytes")

    size = gen_unity_yaml(f"{out}/SampleScene.unity")
    add("SampleScene", "Unity", ".unity", size)
    print(f"  SampleScene.unity        {size:>10,} bytes")

    size = gen_terrain_raw(f"{out}/Terrain.raw", w=513, h=513, seed_x=0.05, seed_y=0.05)
    add("Terrain", "Unity", ".raw", size)
    print(f"  Terrain.raw              {size:>10,} bytes")

    size = gen_terrain_raw(f"{out}/Terrain_large.raw", w=1025, h=1025, seed_x=0.03, seed_y=0.03)
    add("Terrain_large", "Unity", ".raw", size)
    print(f"  Terrain_large.raw        {size:>10,} bytes")

    size = gen_unity_shader(f"{out}/Shaders.shader")
    add("Shaders", "Unity", ".shader", size)
    print(f"  Shaders.shader           {size:>10,} bytes")

    size = gen_yaml_config(f"{out}/k8s_config.yaml")
    add("k8s_config", "Config", ".yaml", size)
    print(f"  k8s_config.yaml          {size:>10,} bytes")

    size = gen_toml_config(f"{out}/services.toml")
    add("services", "Config", ".toml", size)
    print(f"  services.toml            {size:>10,} bytes")

    size = gen_mdix(f"{out}/platform.mdix")
    add("platform", "DixScript", ".mdix", size)
    print(f"  platform.mdix            {size:>10,} bytes")

    size = gen_dll_stub(f"{out}/Assembly.dll")
    add("Assembly", "Binary", ".dll", size)
    print(f"  Assembly.dll             {size:>10,} bytes")

    manifest_path = f"{out}/special_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written -> {manifest_path}")
    print(f"Total files: {len(manifest)}")

if __name__ == "__main__":
    main()
