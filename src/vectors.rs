use core::ops::{Add, Mul, Neg, Sub};
use core::fmt::Debug;
use defmt::error;
use embedded_graphics::prelude::Point;

use micromath::vector::{Component, Vector, Vector2d};

#[cfg(feature = "nalgebra")] 
use nalgebra as na;
#[cfg(feature = "nalgebra")] 
use na::Vector2;

use num_traits::{FromPrimitive, ToPrimitive};
use num_traits::Float;

// use defmt::info;
use thiserror_no_std::Error;

#[derive(Debug, Error, defmt::Format)]
pub enum SpriteError {
    ConversionError,
    Infallible(#[from] core::convert::Infallible),
    OtherError,
}

extern crate micromath as mm;

pub trait VecComp: 
    Component
    + ToPrimitive
    + FromPrimitive
    + core::ops::Neg<Output = Self>
    // + SquareRoot
    + defmt::Format
{}

impl VecComp for i32 {}
impl VecComp for f32 {}


#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpriteVector<C: VecComp>(pub Vector2d<C>);

impl<C: VecComp> SpriteVector<C> {
    fn get_x(&self) -> C {
        self.0.x
    }
    fn get_y(&self) -> C {
        self.0.y
    }
}

impl Mul<f32> for SpriteVector<f32> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let x = self.x * rhs;
        let y = self.y * rhs;
        (x,y).into()
    }
}
impl Mul<f32> for SpriteVector<i32> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let x = self.x as f32 * rhs;
        let y = self.y as f32 * rhs;
        (x,y).into()
    }
}
impl Mul<i32> for SpriteVector<i32> {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self::Output {
        let x = self.x * rhs;
        let y = self.y * rhs;
        (x,y).into()
    }
}

impl Neg for SpriteVector<i32> {
    type Output = SpriteVector<i32>;

    fn neg(self) -> Self::Output {
        (-self.x, -self.y).into()
    }
}
impl Neg for SpriteVector<f32> {
    type Output = SpriteVector<f32>;

    fn neg(self) -> Self::Output {
        (-self.x, -self.y).into()
    }
}
impl<C: VecComp> Add for SpriteVector<C> 
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let x = self.get_x() + rhs.get_x();
        let y = self.get_y() + rhs.get_y();
        (x,y).into()
    }

}

impl<C: VecComp> Sub for SpriteVector<C> 
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let x = self.get_x() - rhs.get_x();
        let y = self.get_y() - rhs.get_y();
        (x,y).into()
    }

}

impl<C: VecComp> From<Point> for SpriteVector<C> {
    fn from(value: Point) -> Self {
        let x = C::from_i32(value.x).unwrap_or_default();
        let y = C::from_i32(value.y).unwrap_or_default();
        Self(Vector2d { x, y })
    }
}

impl From<(f32,f32)> for SpriteVector<i32> {
    fn from(value: (f32,f32)) -> Self {
        let (x,y) = value;
        (Float::round(x) as i32, y.round() as i32).into()
    }
}


impl<C: VecComp> core::ops::DerefMut for SpriteVector<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<C: VecComp> core::ops::Deref for SpriteVector<C> {
    type Target = Vector2d<C>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<C: VecComp> defmt::Format for SpriteVector<C> {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{{ x: {}, y: {} }}", self.0.x, self.0.y)
    }
}

impl<C: VecComp> PartialOrd for SpriteVector<C> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.to_array().partial_cmp(&other.0.to_array())
    }
}


impl<C: VecComp> From<SpriteVector<C>> for Point {
    fn from(value: SpriteVector<C>) -> Self {

        let x = value.0.x.to_f32().unwrap_or_else(||{
            error!("unable to convert from {}", value.0.x);
            panic!();
            // f32::default()
        });
        let y = value.0.y.to_f32().unwrap_or_else(||{
            error!("unable to convert from {}", value.0.y);
            panic!();
            // f32::default()
        });
        Point { x: x.round() as i32, y: y.round() as i32 }
    }
}

impl<C: VecComp> From<(C,C)> for SpriteVector<C> {
    fn from(value: (C,C)) -> Self {
        Self(value.into())
    }
}

impl<C: VecComp>  SpriteVector<C> {
    pub fn new(x: C, y: C) -> Self {
        Self(Vector2d { x, y })
    }
}

impl<C: VecComp> From<Vector2d<C>> for SpriteVector<C> {
    fn from(value: Vector2d<C>) -> Self {
        SpriteVector(value)
    }
}


pub trait VecNormalize {
    fn normalize(&self) -> Self;
}


impl VecNormalize for Point {
    /// Normalize the vector into a unit vector (keep direction, but magnitude is one)
    /// if we would divide by zero return a vector of 0,0
    fn normalize(&self) -> Self {
        let mag = (self.x.pow(2) + self.y.pow(2)).isqrt();
        if mag == 0 {
            Self::zero()
        } else {
            Self { x: self.x / mag, y: self.y / mag }
        }
    }
}


// fn normalize_vector2d_f32(a_vec: micromath::vector::Vector2d<f32>) -> micromath::vector::Vector2d<f32> {
//     let mag = micromath::vector::Vector::magnitude(a_vec);
//     micromath::vector::Vector2d { x: a_vec.x / mag, y: a_vec.y / mag }
// }


pub trait VectorLike2d<C> {
    fn get_x(self) -> C;
    fn get_y(self) -> C;

    /// calculate the dot product of two vectors
    fn dot(self, rhs: Self) -> C;

    /// Compute the magnitude of a vector
    fn magnitude(self) -> f32;

    /// Compute the distance between two vectors
    fn distance(self, rhs: Self) -> f32;

}


pub struct Coefficient(pub f32);

impl From<i32> for Coefficient {
    fn from(value: i32) -> Self {
        Self(value as f32)
    }
}

impl From<f32> for Coefficient {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<Coefficient> for f32 {
    fn from(value: Coefficient) -> Self {
        value.0
    }
}


// #[derive(Clone, Copy, Debug, PartialEq, Default)]
// pub struct VelocityCalculationResults<T: VecComp> {
//     pub initial: SpriteVector<T>,
//     pub collision_normal: SpriteVector<T>,
//     pub normal: SpriteVector<T>,
//     pub perpendicular: SpriteVector<T>,
//     pub parallel: SpriteVector<T>,
//     pub reflected: SpriteVector<T>,
// }

// impl<T: VecComp> defmt::Format for VelocityCalculationResults<T> {
//     fn format(&self, fmt: defmt::Formatter) {
//         defmt::write!(fmt, "vcr: i:{} cn:{} n:{} pe:{} pa:{} r:{}",
//         self.initial,
//         self.collision_normal,
//         self.normal,
//         self.perpendicular,
//         self.parallel,
//         self.reflected)
//     }
// }

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpriteVectorFloat(pub Vector2d<f32>);

impl PartialOrd for SpriteVectorFloat {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.to_array().partial_cmp(&other.0.to_array())
    }
}

impl From<SpriteVectorFloat> for SpriteVector<i32> {
    fn from(value: SpriteVectorFloat) -> Self {
        (value.x, value.y).into()
    }
}

impl From<SpriteVectorFloat> for SpriteVector<f32> {
    fn from(value: SpriteVectorFloat) -> Self {
        (value.x, value.y).into()
    }
}

impl From<(f32,f32)> for SpriteVectorFloat {
    fn from(value: (f32,f32)) -> Self {
        Self(value.into())
    }
}

impl From<Point> for SpriteVectorFloat {
    fn from(value: Point) -> Self {
        Self((value.x as f32, value.y as f32).into())
    }
}

impl From<SpriteVectorFloat> for Point {
    fn from(value: SpriteVectorFloat) -> Self {
        Point::new(value.x.round() as i32, value.y.round() as i32)
    }
}
impl core::ops::Deref for SpriteVectorFloat {
    type Target = Vector2d<f32>;
    
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for SpriteVectorFloat {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl defmt::Format for SpriteVectorFloat {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{{ x: {}, y: {} }}", self.0.x, self.0.y)
    }
}

impl From<(i32,i32)> for SpriteVectorFloat {
    fn from(value: (i32,i32)) -> Self {
        (value.0 as f32, value.1 as f32).into()
    }
}

impl Neg for SpriteVectorFloat {
    type Output = SpriteVectorFloat;

    fn neg(self) -> Self::Output {
        (-self.x, -self.y).into()
    }
}

impl Sub for SpriteVectorFloat {
    type Output = SpriteVectorFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        (self.x - rhs.x, self.y - rhs.y).into()
    }
}

impl Add for SpriteVectorFloat {
    type Output = SpriteVectorFloat;

    fn add(self, rhs: Self) -> Self::Output {
        (self.x + rhs.x, self.y + rhs.y).into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpriteVectorInt(pub Vector2d<i32>);


impl From<(i32,i32)> for SpriteVectorInt {
    fn from(value: (i32,i32)) -> Self {
        Self(value.into())
    }
}

impl From<Point> for SpriteVectorInt {
    fn from(value: Point) -> Self {
        Self((value.x, value.y).into())
    }
}

impl core::ops::DerefMut for SpriteVectorInt {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl core::ops::Deref for SpriteVectorInt {
    type Target = Vector2d<i32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl defmt::Format for SpriteVectorInt {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{{ x: {}, y: {} }}", self.0.x, self.0.y)
    }
}

impl From<SpriteVectorInt> for SpriteVectorFloat {
    fn from(value: SpriteVectorInt) -> Self {
        (value.x as f32, value.y as f32).into()
    }
}
impl From<SpriteVectorFloat> for SpriteVectorInt {
    fn from(value: SpriteVectorFloat) -> Self {
        (value.x.round() as i32, value.y.round() as i32).into()
    }
}

impl From<SpriteVector<f32>> for SpriteVectorFloat {
    fn from(value: SpriteVector<f32>) -> Self {
        (value.x, value.y).into()
    }
}

impl Mul<Coefficient> for SpriteVectorFloat {
    type Output = SpriteVectorFloat;

    fn mul(self, rhs: Coefficient) -> Self::Output {
        let x = self.x * rhs.0;
        let y = self.y * rhs.0;
        (x,y).into()
    }
}

impl Mul<Coefficient> for SpriteVectorInt {
    type Output = SpriteVectorInt;

    fn mul(self, rhs: Coefficient) -> Self::Output {
        let x = self.x * rhs.0 as i32;
        let y = self.y * rhs.0 as i32;
        (x,y).into()
    }
}


pub trait Velocity: Copy + Clone + PartialEq + defmt::Format + Add<Output = Self> + Neg<Output = Self> + Sub<Output = Self>
where 
    Self: From<Point> + From<(i32,i32)> + From<(f32,f32)> + Mul<Coefficient, Output = Self>, 
{
    /// calculate the dot product of two vectors
    fn dot(self, rhs: Self) -> f32;

    /// Compute the magnitude of a vector
    fn magnitude(self) -> f32;

    // /// Compute the distance between two vectors
    // fn distance(self, rhs: Self) -> f32;

    #[inline]
    fn speed(&self) -> f32 {
        self.magnitude()
    }

    #[inline]
    fn direction(&self) -> Self {
        self.normalize()
    }

    fn zero() -> Self;

    fn get_x(self) -> f32;
    fn get_y(self) -> f32;

    /// Normalize the vector into a unit vector (keep direction, but magnitude is one)
    /// try not to divide by zero 
    fn normalize(&self) -> Self;

    /// rotate the vector 90 degrees
    /// set x to negative y , and y to x
    fn rotate90(self) -> Self
    {
        let x = -self.get_y();
        let y = self.get_x();
        (x, y).into()
    }


    /// Calculate the reflection off of the surface in the collision, given the surface_normal
    /// 
    fn calculate_reflection(&self, collision_normal: &Self) -> Self {
        let velocity = self;

        let speed = velocity.speed();
        let direction = velocity.direction();

        let surface_normal_speed = collision_normal.speed();
        let surface_normal_directon = collision_normal.direction();

        // Calculate the component of the incoming velocity perpendicular to the collision surface
        // this is the direction of the colision normal scaled by 
        // the dot product of self and the direction of the collision normal
        let v_perpendicular = {
            surface_normal_directon * Coefficient(velocity.dot(surface_normal_directon))
        };

        // Calculate the component of the incoming velocity parallel to the collision surface
        // this is simply self minus the perpendicular velocity
        let v_parallel = *velocity - v_perpendicular;

        // The reflected perpendicular velocity is reversed and scaled by the COR
        let reflected_perpendicular = -v_perpendicular;
        
        // The reflected velocity is the sum of the reflected perpendicular and parallel components
        let mut reflected = reflected_perpendicular + v_parallel;
        reflected = reflected.normalize() * Coefficient(speed);

        {
            defmt::debug!("  initial_velocity: {}, s: {}, d: {}", velocity ,speed, direction);
            defmt::debug!("  surface_normal:   {}, s: {}, d: {}", collision_normal, surface_normal_speed, surface_normal_directon);
            defmt::debug!("  v_perp: {}, v_parallel: {}", v_perpendicular, v_parallel);
            defmt::debug!("  reflected_v: {}, s: {}, d: {}", reflected, reflected.speed(), reflected.direction());
        }

        reflected
    }

    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{{ x: {}, y: {} }}", self.get_x(), self.get_y())
    }

}


impl Velocity for SpriteVectorFloat {
    fn dot(self, rhs: Self) -> f32 {
        self.0.dot(rhs.0)
    }

    fn magnitude(self) -> f32 {
        self.0.magnitude()
    }

    fn zero() -> Self {
        Self(Vector2d::<f32>::from((0_f32,0_f32)))
    }

    fn normalize(&self) -> Self {
        let mag =  self.0.magnitude();
        if mag as i32 == 0 {
            Velocity::zero()
        } else {
            let x = self.x / mag;
            let y = self.y / mag;
            (x, y).into()
        }
    }

    fn rotate90(self) -> Self {
        let x = -self.y;
        let y = self.x;
        (x, y).into()
    }
    
    fn get_x(self) -> f32 {
        self.x
    }
    
    fn get_y(self) -> f32 {
        self.y
    }
}

#[cfg(feature = "nalgebra")]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpriteVectorFloatNa(pub Vector2<f32>);

#[cfg(feature = "nalgebra")]
impl core::ops::DerefMut for SpriteVectorFloatNa {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(feature = "nalgebra")]
impl core::ops::Deref for SpriteVectorFloatNa {
    type Target = Vector2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(feature = "nalgebra")]
impl PartialOrd for SpriteVectorFloatNa {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

#[cfg(feature = "nalgebra")]
impl From<SpriteVectorFloatNa> for SpriteVector<i32> {
    fn from(value: SpriteVectorFloatNa) -> Self {
        (value.x, value.y).into()
    }
}

#[cfg(feature = "nalgebra")]
impl From<SpriteVectorFloatNa> for SpriteVector<f32> {
    fn from(value: SpriteVectorFloatNa) -> Self {
        (value.x, value.y).into()
    }
}

#[cfg(feature = "nalgebra")]
impl From<(f32,f32)> for SpriteVectorFloatNa {
    fn from(value: (f32,f32)) -> Self {
        Self(Vector2::new(value.0, value.1))
    }
}

#[cfg(feature = "nalgebra")]
impl From<Point> for SpriteVectorFloatNa {
    fn from(value: Point) -> Self {
        Self(Vector2::new(value.x as f32, value.y as f32))
    }
}

#[cfg(feature = "nalgebra")]
impl From<SpriteVectorFloatNa> for Point {
    fn from(value: SpriteVectorFloatNa) -> Self {
        Point::new(value.x.round() as i32, value.y.round() as i32)
    }
}

#[cfg(feature = "nalgebra")]
impl defmt::Format for SpriteVectorFloatNa {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{{ x: {}, y: {} }}", self.0.x, self.0.y)
    }
}

#[cfg(feature = "nalgebra")]
impl From<(i32,i32)> for SpriteVectorFloatNa {
    fn from(value: (i32,i32)) -> Self {
        (value.0 as f32, value.1 as f32).into()
    }
}

#[cfg(feature = "nalgebra")]
impl Neg for SpriteVectorFloatNa {
    type Output = SpriteVectorFloatNa;

    fn neg(self) -> Self::Output {
        (-self.x, -self.y).into()
    }
}

#[cfg(feature = "nalgebra")]
impl Sub for SpriteVectorFloatNa {
    type Output = SpriteVectorFloatNa;

    fn sub(self, rhs: Self) -> Self::Output {
        (self.x - rhs.x, self.y - rhs.y).into()
    }
}

#[cfg(feature = "nalgebra")]
impl Add for SpriteVectorFloatNa {
    type Output = SpriteVectorFloatNa;

    fn add(self, rhs: Self) -> Self::Output {
        (self.x + rhs.x, self.y + rhs.y).into()
    }
}

#[cfg(feature = "nalgebra")]
impl Mul<Coefficient> for SpriteVectorFloatNa {
    type Output = Self;

    fn mul(self, rhs: Coefficient) -> Self::Output {
        (self.x * rhs.0, self.y * rhs.0).into()
    }
}

#[cfg(feature = "nalgebra")]
impl Velocity for SpriteVectorFloatNa {
    fn dot(self, rhs: Self) -> f32 {
        self.0.dot(&rhs.0)
    }

    fn magnitude(self) -> f32 {
        self.0.magnitude()
    }

    fn zero() -> Self {
        Self(Vector2::zeros())
    }

    fn normalize(&self) -> Self {
        Self(self.0.normalize())
    }

    fn rotate90(self) -> Self {
        let x = -self.0.y;
        let y = self.0.x;
        (x, y).into()
    }
    
    fn get_x(self) -> f32 {
        self.x
    }
    
    fn get_y(self) -> f32 {
        self.y
    }
}


#[cfg(feature = "nalgebra")]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpriteVectorIntNa(pub Vector2<i32>);

#[cfg(feature = "nalgebra")]
impl PartialOrd for SpriteVectorIntNa {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

#[cfg(feature = "nalgebra")]
impl Neg for SpriteVectorIntNa {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

#[cfg(feature = "nalgebra")]
impl Sub for SpriteVectorIntNa {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

#[cfg(feature = "nalgebra")]
impl Add for SpriteVectorIntNa {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

#[cfg(feature = "nalgebra")]
impl SpriteVectorIntNa {
    pub fn new(x: i32, y: i32) -> Self {
        Self(Vector2::new(x, y))
    }
}

#[cfg(feature = "nalgebra")]
impl core::ops::DerefMut for SpriteVectorIntNa {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(feature = "nalgebra")]
impl core::ops::Deref for SpriteVectorIntNa {
    type Target = Vector2<i32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(feature = "nalgebra")]
impl Mul<Coefficient> for SpriteVectorIntNa {
    type Output = SpriteVectorIntNa;

    fn mul(self, rhs: Coefficient) -> Self::Output {
        let x = self.x as f32 * rhs.0;
        let y = self.y as f32 * rhs.0;

        Self::new(x.round() as i32, y.round() as i32)
    }
}

#[cfg(feature = "nalgebra")]
impl From<(i32, i32)> for SpriteVectorIntNa {
    fn from(value: (i32, i32)) -> Self {
        Self::new(value.0, value.1)
    }
}

#[cfg(feature = "nalgebra")]
impl From<(f32, f32)> for SpriteVectorIntNa {
    fn from(value: (f32, f32)) -> Self {
        let (x,y) = value;
        Self::new(x.round() as i32, y.round() as i32)
    }
}

#[cfg(feature = "nalgebra")]
impl From<Point> for SpriteVectorIntNa {
    fn from(value: Point) -> Self {
        Self::new(value.x, value.y)
    }
}

#[cfg(feature = "nalgebra")]
impl From<SpriteVectorIntNa> for Point {
    fn from(value: SpriteVectorIntNa) -> Self {
        Point::new(value.x, value.y)
    }
}

#[cfg(feature = "nalgebra")]
impl defmt::Format for SpriteVectorIntNa {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{{ x: {}, y: {} }}", self.0.x, self.0.y)
    }
}

#[cfg(feature = "nalgebra")]
impl Velocity for SpriteVectorIntNa {
    fn dot(self, rhs: Self) -> f32 {
        self.0.dot(&rhs.0) as f32
    }

    fn magnitude(self) -> f32 {
        let mag = (self.x.pow(2) + self.y.pow(2)).isqrt();
        mag as f32
    }

    fn zero() -> Self {
        Self::new(0, 0)
    }

    fn get_x(self) -> f32 {
        self.x as f32
    }

    fn get_y(self) -> f32 {
        self.y as f32
    }

    fn normalize(&self) -> Self {
        let mag = (self.x.pow(2) + self.y.pow(2)).isqrt();
        if mag == 0 {
            Self::zero()
        } else {
            Self::new( self.x / mag, self.y / mag )
        }        
    }
}
