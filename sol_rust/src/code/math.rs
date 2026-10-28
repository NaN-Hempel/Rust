// Contains mathematical functions and constants for the simulation.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

pub use crate::code::physics::*;

pub const PI: f64 = 3.1415926535897932385;

pub fn deg_to_rad(degrees: f64) -> f64 {
    degrees * (PI / 180.0)
}


// Coefficients of Yoshida's fourth-order integrator (H. Yoshida, 1990), built from three
// leapfrog steps of lengths w1, w0, w1, with w0 = -2^(1/3) * w1 so the second-order errors cancel.
// W0 is negative: the middle step briefly goes backwards in time.
const CBRT_2: f64 = 1.259_921_049_894_873_2; // 2^(1/3)
const W1: f64 = 1.0 / (2.0 - CBRT_2);
const W0: f64 = -CBRT_2 * W1;
// Drift (position) and kick (velocity) fractions of dt.
pub const DRIFT: [f64; 4] = [W1 / 2.0, (W0 + W1) / 2.0, (W0 + W1) / 2.0, W1 / 2.0];
pub const KICK: [f64; 3] = [W1, W0, W1];

// One fourth-order Yoshida step for bodies whose acceleration depends only on their positions
// (x'' = a(x)), as with gravity. Updates positions and velocities in place.
// accel fills its second argument with the accelerations at the given positions; acc is
// scratch space for it, the same length as positions.
// Being symplectic, it doesn't slowly gain or lose energy over long runs the way RK4 does,
// and it needs 3 acceleration evaluations per step instead of RK4's 4.
pub fn yoshida4_step(positions: &mut [Vec3], velocities: &mut [Vec3], dt: f64, acc: &mut [Vec3], accel: impl Fn(&[Vec3], &mut [Vec3])) {
    for stage in 0..4 {
        for (x, &v) in positions.iter_mut().zip(velocities.iter()) {
            *x += v * (DRIFT[stage] * dt);
        }
        if stage < 3 {
            accel(positions, acc);
            for (v, &a) in velocities.iter_mut().zip(acc.iter()) {
                *v += a * (KICK[stage] * dt);
            }
        }
    }
}

// A 3D vector, used for positions, velocities and forces.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    // Length of the vector.
    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    // Length squared; cheaper than norm() as it needs no square root.
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn dot(self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)
    }

    // The same direction with length 1.
    pub fn normalised(self) -> Vec3 {
        self / self.norm()
    }

    // Rotated by angle (radians) about the unit vector axis, anticlockwise looking down the axis
    // (Rodrigues' rotation formula).
    pub fn rotated(self, axis: Vec3, angle: f64) -> Vec3 {
        let (sin, cos) = angle.sin_cos();
        self * cos + axis.cross(self) * sin + axis * (axis.dot(self) * (1.0 - cos))
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        *self = *self + other;
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Vec3) {
        *self = *self - other;
    }
}
