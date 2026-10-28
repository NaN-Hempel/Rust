//tester file

fn fast_inv_sqrt(number: f32) -> f32 {
    const THREEHALFS: f32 = 1.5;
    let mut i: i32 = number.to_bits() as i32;
    i = 0x5f3759df - (i >> 1);
    let mut x: f32 = f32::from_bits(i as u32);
    x = x * (THREEHALFS - 0.5 * number * x * x);
    x
}

fn main() {
    let number: f32 = 1.0;
    let inv_sqrt = fast_inv_sqrt(number);
    println!("Fast inverse square root of {} is approximately {}", number, inv_sqrt);
}