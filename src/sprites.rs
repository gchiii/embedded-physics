use core::f32::consts::PI;
use core::ops::Neg;

use defmt::info;
use embedded_graphics::prelude::{Angle, ContainsPoint, Dimensions, DrawTarget, PixelColor, Point, PointsIter, Transform};
use embedded_graphics::primitives::{self, PrimitiveStyle, StyledDrawable};
use embedded_graphics::primitives::{
    Line,
    Rectangle,
    Circle,
    // Ellipse,
    // Arc,
    // Sector,
    Triangle,
    Polyline,
    // RoundedRectangle
};
use embedded_graphics::Drawable;
use heapless::Vec;
use num_traits::{Float, ToPrimitive};

use crate::geometry::{Area, ClosestEdge, ClosestPoint, PointExt, SurfaceNormal};
use crate::vectors::{GfxVector, VecComp, VecNormalize};
 

/// Make an enum that serves as a wrapper around the various Primitives from the embedded-graphics library
/// 
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, defmt::Format)]
pub enum SpritePrimitive<'a> 
{
    Line(Line),
    Circle(Circle),
    Rectangle(Rectangle),
    Polyline(Polyline<'a>),
    Triangle(Triangle),
}


impl<'a> ContainsPoint for SpritePrimitive<'a> {
    fn contains(&self, point: Point) -> bool {
        match self {
            SpritePrimitive::Line(line) => line.points().any(|p| p == point), 
            SpritePrimitive::Circle(circle) => circle.contains(point),
            SpritePrimitive::Rectangle(rectangle) => rectangle.contains(point),
            SpritePrimitive::Polyline(polyline) => polyline.points().any(|p| p == point),
            SpritePrimitive::Triangle(triangle) => triangle.contains(point),
        }
    }
}

impl<'a> Default for SpritePrimitive<'a> {
    fn default() -> Self {
        SpritePrimitive::Circle(Circle::default())
    }
}

impl<'a> SpritePrimitive<'a> {
    /// center point of this primitive
    pub fn center(&self) -> Point {
        match self {
            SpritePrimitive::Line(line) => line.bounding_box().center(),
            SpritePrimitive::Circle(circle) => circle.center(),
            SpritePrimitive::Rectangle(rectangle) => rectangle.center(),
            SpritePrimitive::Polyline(polyline) => polyline.bounding_box().center(),
            SpritePrimitive::Triangle(triangle) => triangle.bounding_box().center(),
        }
    }

    /// rough estimate of distance between the bounding boxes of this and other
    pub fn box_distance(&self, other: &Self) -> i32 {
        let area_of_distance = self.center().distance_squared(other.center());
        let area_of_objects = self.area() + other.area();
        area_of_distance - area_of_objects as i32
    }

    pub fn perimeter_point_near(&self, point: Point) -> Point {
        match self {
            SpritePrimitive::Line(line) => line.closest_point(point),
            SpritePrimitive::Circle(circle) => {
                let delta = circle.center() - point;
                delta.normalize() * (circle.diameter as i32 / 2)                
            },
            SpritePrimitive::Rectangle(rectangle) => rectangle.closest_edge(point).closest_point(point),
            SpritePrimitive::Polyline(polyline) => polyline.closest_edge(point).closest_point(point),
            SpritePrimitive::Triangle(triangle) => triangle.closest_edge(point).closest_point(point),
        }
    }

    /// iterate through perimeter of this primitive, and check if rhs primitive contains the point
    pub fn overlaps(&self, rhs: &Self) -> bool {
        false
    }
}


impl<'a, C: PixelColor> StyledDrawable<PrimitiveStyle<C>> for SpritePrimitive<'a> {
    type Color = C;

    type Output = ();

    fn draw_styled<D>(&self, style: &PrimitiveStyle<C>, target: &mut D) -> Result<Self::Output, D::Error>
    where
        D: DrawTarget<Color = Self::Color> {
        match self {
            SpritePrimitive::Line(line) => line.draw_styled(style, target),
            SpritePrimitive::Circle(circle) => circle.draw_styled(style, target),
            SpritePrimitive::Rectangle(rectangle) => rectangle.draw_styled(style, target),
            SpritePrimitive::Polyline(polyline) => polyline.draw_styled(style, target),
            SpritePrimitive::Triangle(triangle) => triangle.draw_styled(style, target),
        }
    }
}

impl<'a> Area for SpritePrimitive<'a> {
    fn area(&self) -> u32 {
        match self {
            SpritePrimitive::Line(_line) => 0,
            SpritePrimitive::Rectangle(rectangle) => rectangle.size.area(),
            SpritePrimitive::Circle(circle) => {
                        // Area = (π/4) × d^2, where 'd' is the diameter
                        let d_squared = circle.diameter * circle.diameter;
                        let area = (PI/4.0) * d_squared as f32;
                        area as u32
                    },
            SpritePrimitive::Polyline(polyline) => {
                        // just use the bounding box for now, we can revisit later
                        // polyline.points().map(|p| info!("{}", p));
                        polyline.bounding_box().size.area()
                    },
            SpritePrimitive::Triangle(triangle) => {
                let (p1, p2, p3) = triangle.vertices.into();
                ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)).unsigned_abs() / 2
            },
        }
    }
}

impl<'a> Dimensions for SpritePrimitive<'a> {
    fn bounding_box(&self) -> primitives::Rectangle {
        match self {
            SpritePrimitive::Line(line) => line.bounding_box(),
            SpritePrimitive::Circle(circle) => circle.bounding_box(),
            SpritePrimitive::Rectangle(rectangle) => rectangle.bounding_box(),
            SpritePrimitive::Polyline(polyline) => polyline.bounding_box(),
            SpritePrimitive::Triangle(triangle) => triangle.bounding_box(),
        }
    }
}

impl<'a> From<primitives::Line> for SpritePrimitive<'a> {
    fn from(value: primitives::Line) -> Self {
        Self::Line(value)
    }
}
impl<'a> From<primitives::Circle> for SpritePrimitive<'a> {
    fn from(value: primitives::Circle) -> Self {
        Self::Circle(value)
    }
}
impl<'a> From<primitives::Rectangle> for SpritePrimitive<'a> {
    fn from(value: primitives::Rectangle) -> Self {
        Self::Rectangle(value)
    }
}
impl<'a> From<primitives::Triangle> for SpritePrimitive<'a> {
    fn from(value: primitives::Triangle) -> Self {
        Self::Triangle(value)
    }
}
impl<'a> From<primitives::Polyline<'a>> for SpritePrimitive<'a> {
    fn from(value: primitives::Polyline<'a>) -> Self {
        Self::Polyline(value)
    }
}

impl<'a> Transform for SpritePrimitive<'a> {
    fn translate(&self, by: Point) -> Self {
        match self {
            SpritePrimitive::Line(line) => line.translate(by).into(),
            SpritePrimitive::Circle(circle) => circle.translate(by).into(),
            SpritePrimitive::Rectangle(rectangle) => rectangle.translate(by).into(),
            SpritePrimitive::Polyline(polyline) => polyline.translate(by).into(),
            SpritePrimitive::Triangle(triangle) => triangle.translate(by).into(),
        }
    }

    fn translate_mut(&mut self, by: Point) -> &mut Self {
        *self = self.translate(by);
        self
    }
}

impl<'a> SurfaceNormal for SpritePrimitive<'a> {
    fn surface_normal(&self, point: impl Into<Point> + Copy) -> Point {
        match self {
            SpritePrimitive::Line(line) => line.surface_normal(point),
            SpritePrimitive::Circle(circle) => circle.surface_normal(point),
            SpritePrimitive::Rectangle(rectangle) => rectangle.surface_normal(point),
            SpritePrimitive::Polyline(polyline) => polyline.surface_normal(point),
            SpritePrimitive::Triangle(triangle) => triangle.surface_normal(point),
        }
    }
    
    fn distance(&self, point: impl Into<Point> + Copy) -> f32 {
        match self {
            SpritePrimitive::Line(line) => line.distance(point),
            SpritePrimitive::Circle(circle) => circle.distance(point),
            SpritePrimitive::Rectangle(rectangle) => rectangle.distance(point),
            SpritePrimitive::Polyline(polyline) => polyline.distance(point),
            SpritePrimitive::Triangle(triangle) => triangle.distance(point),
        }
    }
}


#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, defmt::Format)]
pub struct Sprite<'a, C: PixelColor, T: VecComp>
{
    name: &'a str,
    style: PrimitiveStyle<C>,
    line_style: Option<PrimitiveStyle<C>>,
    shape: SpritePrimitive<'a>,
    area: u32,
    velocity: GfxVector<T>,
}

impl<'a, C: PixelColor, T: VecComp> Drawable for Sprite<'a, C, T> {
    type Color = C;

    type Output = ();

    fn draw<D>(&self, target: &mut D) -> Result<Self::Output, D::Error>
    where
        D: DrawTarget<Color = Self::Color> {
        self.shape.draw_styled(&self.style, target)?;
        if let Some(line_style) = self.line_style {
            let center: Point = self.shape.center();
            let delta: Point = self.velocity().into();
            Line::with_delta(center, delta).draw_styled(&line_style, target)?;
        }
        Ok(())
    }
}

impl<'a, C: PixelColor, T: VecComp> Default for Sprite<'a, C, T> {
    fn default() -> Self {
        let shape = SpritePrimitive::default();
        let area = shape.area();
        Self { 
            name: Default::default(), 
            style: Default::default(), 
            line_style: Default::default(), 
            shape, 
            area, 
            velocity: Default::default() 
        }
    }
}

impl<'a, C: PixelColor, T: VecComp> Sprite<'a, C, T> {
    pub fn new(name: &'a str, shape: impl Into<SpritePrimitive<'a>>) -> Self {
        let shape: SpritePrimitive<'a> = shape.into();
        let area = shape.area();
        Self {
            name,
            shape,
            area,
            ..Default::default()
        }
    }

    pub fn with_style(self, style: PrimitiveStyle<C>) -> Self {
        Self {
            style,
            ..self
        }
    }

    pub fn with_line_style(self, line_style: PrimitiveStyle<C>) -> Self {
        Self {
            line_style: Some(line_style),
            ..self
        }
    }

    pub fn with_velocity(self, velocity: GfxVector<T>) -> Self {
        Self {
            velocity,
            ..self
        }
    }
    
    pub fn velocity_mut(&mut self) -> &mut GfxVector<T> {
        &mut self.velocity
    }
    
    pub fn set_velocity(&mut self, velocity: GfxVector<T>) {
        self.velocity = velocity;
    }
    
    pub fn velocity(&self) -> GfxVector<T> {
        self.velocity
    }
    
    pub fn shape(&self) -> SpritePrimitive<'a> {
        self.shape
    }
    
}


// Sized + Neg<Output = GfxVector<T>> + Sub<Output = GfxVector<T>> + Add<Output = GfxVector<T>>
impl<'a, C: PixelColor, T: VecComp> Sprite<'a, C, T> 
where 
    // GfxVector<T>:  for<'b> Sub<&'b GfxVector<T>, Output = GfxVector<T>>, 
    // for<'b> &'b GfxVector<T>: Sub<&'b GfxVector<T>, Output = GfxVector<T>> + Mul<T, Output = GfxVector<T>>, 
    // f32: Mul<GfxVector<T>, Output = GfxVector<T>> 
{
    pub fn next_position(&self) -> SpritePrimitive<'_> {
        self.shape.translate(self.velocity.into())
    }

    pub fn area(&self) -> u32 {
        self.area
    }
    
    pub fn is_moving(&self) -> bool {
        !(self.velocity.x.to_i32().unwrap() == 0 && self.velocity.y.to_i32().unwrap() == 0)
    }

    pub fn distance_between(&self, other: &Self) -> i32 {
        self.shape.box_distance(&other.shape)
    }

    pub fn about_to_collide(&self, other: &Self) -> bool {
        let next_self = self.shape.translate(self.velocity.into());
        let next_other = other.shape.translate(other.velocity.into());
        next_self.box_distance(&next_other) < 1
    }
    
    pub fn is_collision(sprite1: &Sprite<'a, C, T>, sprite2: &Sprite<'a, C, T>) -> bool {
        let distance = sprite1.distance_between(sprite2);
        distance < 1
    }

    pub fn update_velocity(&mut self, other: &Self) -> bool {
        let mut bounced = false;
        if Self::is_collision(self, other) && self.is_moving() {
            let collision_normal = other.shape.surface_normal(self.shape.center());
            let collision_normal = GfxVector::from(collision_normal);
            // let velocity = self.velocity_mut();

            let velocity = self.velocity.calculate_reflection_vector(&collision_normal);
            self.set_velocity(velocity);
            bounced = true;
        }
        bounced
    }
    
    pub fn set_direction_from_angle(&mut self, direction: Angle) {
        let radius = self.half_width().to_f32().unwrap() + 1.0;
        let angle = direction.to_radians();
        let (mut x, mut y) = Float::sin_cos(angle);
        x *= radius;
        y *= radius;
        // self.set_velocity(GfxVec((x, y).into()));
        self.set_velocity(Point::new(x.to_i32().unwrap(), y.to_i32().unwrap()).into());
    }

    fn half_width(&self) -> u32 {
        self.shape.bounding_box().size.width / 2
    }

    /// move object by applying the direction vector, while checking against containing rectangle
    pub fn move_object_bounded(&mut self, boundary: & impl Dimensions) {
        let mut delta = self.velocity();
        if self.is_moving() {
            let center = self.shape.center();
            let width = self.half_width() as i32;
    
            // check against bounds
            let r = boundary.bounding_box();
            let top_left = r.top_left;
            let bottom_right = Point::new(top_left.x + r.size.width as i32, top_left.y + r.size.height as i32);
    
            if (center.x - top_left.x) <= width || (bottom_right.x - center.x) <= width {
                // bounce off x axis
                delta.x = -delta.x;
            }
            if (center.y - top_left.y) <= width || (bottom_right.y - center.y) <= width {
                // bounce off y axis
                delta.y = -delta.y;
            }
    
            self.set_velocity(delta);
            self.shape.translate_mut(delta.into());
        }
    }

    /// apply the velocity to the position of the shape (ignoring any collisions)
    pub fn move_object(&mut self) {
        if self.is_moving() {
            let delta = self.velocity();
            self.shape.translate_mut(delta.into());
        }
    }
    

}



// #[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
#[derive(Clone, Debug, Default, defmt::Format)]
pub struct SpriteContainer<'a, C: PixelColor, T> 
    where 
        C: PixelColor, 
        T: VecComp + Neg<Output = T> 
{
    boundary: Rectangle,
    sprites: Vec<Sprite<'a, C, T>, 10>,
    _phantom: core::marker::PhantomData<C>,
}

impl<'a, C: PixelColor, T> SpriteContainer<'a, C, T> 
where 
    C: PixelColor, 
    T: VecComp
{
    pub fn new(boundary: Rectangle) -> Self {
        let sprites = Vec::new();
        Self { boundary, sprites, _phantom: core::marker::PhantomData }
    }

    pub fn add_sprite(&mut self, sprite: Sprite<'a, C, T>) -> Result<(), Sprite<'a, C, T>> {
        self.sprites.push(sprite)
    }

    pub fn update_positions(&mut self) {
        let max_idx = self.sprites.len();
        for i in 0..max_idx {
            let (left, right) = self.sprites.split_at_mut(i);
            if let Some((current_sprite, right)) = right.split_first_mut() {
                // info!("sprite: {} vel: {}", current_sprite.name, current_sprite.velocity());
                if current_sprite.is_moving() {
                    // Iterate over the remaining parts to find other elements matching the predicate
                    for other_sprite in left.iter().chain(right.iter()) {
                        if current_sprite.about_to_collide(other_sprite) {
                            if current_sprite.update_velocity(other_sprite) {
                                // info!("{} bounced off of {}", current_sprite.name(), other_sprite.name());
                            }
                        }
                    }
                }
            }
        }
        for current_sprite in self.sprites.iter_mut() {
            if current_sprite.is_moving() {
                current_sprite.move_object_bounded(&self.boundary);
            }
        }
    }
}

impl<'a, C: PixelColor> SpriteContainer<'a, C, i32> 
{
}

impl<'a, C: PixelColor, T> Drawable for SpriteContainer<'a, C, T> 
    where 
        C: PixelColor, 
        T: VecComp + Neg<Output = T> 
{
    type Color = C;

    type Output = ();

    fn draw<D>(&self, target: &mut D) -> Result<Self::Output, D::Error>
    where
        D: DrawTarget<Color = Self::Color> {
        for sprite in self.sprites.iter() {
            sprite.draw(target)?;
        }
        Ok(())
    }
}

