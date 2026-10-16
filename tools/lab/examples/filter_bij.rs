// Scratch: is undo_filter(apply_filter(x)) == x for arbitrary bytes? Adversarial inputs.
use mbfa::filters::*;
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; self.0 = x; x.wrapping_mul(0x2545F4914F6CDD1D) }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}
fn main() {
    let skew_x86: &[u8] = &[0xE8, 0xE9, 0x00, 0xFF, 0x00, 0xFF, 0x01, 0x80, 0x25, 0x8B, 0x0F, 0x85, 0xFE, 0x7F];
    let flags: &[(&str, u8)] = &[
        ("x86 BCJ (9)", FILTER_BCJ), ("ARM (11)", FILTER_BCJ_ARM), ("ARM64 (12)", FILTER_BCJ_ARM64),
        ("PPC (13)", FILTER_BCJ_PPC), ("SPARC (14)", FILTER_BCJ_SPARC), ("RISC-V (15)", FILTER_BCJ_RISCV),
        ("delta1 (1)", FILTER_DELTA1), ("delta2 (2)", FILTER_DELTA2), ("delta3 (3)", FILTER_DELTA3),
        ("delta4 (4)", FILTER_DELTA4), ("shuffle4+delta (7)", FILTER_SHUFFLE4_DELTA),
    ];
    for &(name, flag) in flags {
        let mut rng = Rng(0xC0FFEE ^ flag as u64);
        let (mut bad_skew, mut bad_uni, mut total) = (0u32, 0u32, 0u32);
        let mut example: Option<(usize, usize)> = None;
        for trial in 0..4000 {
            let len = 5 + rng.below(400);
            let skewed = trial % 2 == 0;
            let data: Vec<u8> = (0..len).map(|_| if skewed { skew_x86[rng.below(skew_x86.len())] } else { (rng.next() >> 56) as u8 }).collect();
            let f = apply_filter(&data, flag);
            let r = undo_filter(&f, flag);
            total += 1;
            if r != data {
                if skewed { bad_skew += 1 } else { bad_uni += 1 }
                if example.is_none() { example = Some((len, (0..data.len().min(r.len())).find(|&i| data[i] != r[i]).unwrap_or(0))); }
            }
        }
        eprintln!("{:<20} non-invertible: skewed {:>4}/{}  uniform {:>4}/{}  {}", name, bad_skew, total/2, bad_uni, total/2,
            example.map(|(l,i)| format!("(e.g. len {l}, first diff at byte {i})")).unwrap_or_default());
    }
}
