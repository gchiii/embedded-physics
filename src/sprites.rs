use core::default;
use core::f32::consts::PI;
use core::marker::PhantomData;
use core::ops::{AddAssign, Neg};

use defmt::{error, info};
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
use micromath::vector::{Vector, Vector2d};
use num_traits::{Float, ToPrimitive};

use crate::geometry::{Area, ClosestEdge, ClosestPoint, PointExt, SurfaceNormal};
use crate::vectors::{self, SpriteVector, SpriteVectorFloat, SpriteVectorInt, VVelocity, VecComp, VecNormalize, VectorLike2d, VectorXY, Velocity, VelocityCalculationResults};
 

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
            SpritePrimitive::Triangle(triangle) => {
                let sum_x: i32 = triangle.vertices.iter()
                    .map(|p| p.x)
                    .sum();
                let sum_y: i32 = triangle.vertices.iter()
                    .map(|p| p.y)
                    .sum();
                Point::new(sum_x/3, sum_y/3)
                // triangle.bounding_box().center()
            },
        }
    }

    // fn edge_distance_squared(&self, point: Point) -> i32 {
    //     match self {
    //         SpritePrimitive::Line(line) => {
    //             let p1 = line.closest_point(point);
    //             p1.distance_squared(point)
    //         },
    //         SpritePrimitive::Circle(circle) => {
    //             let center_distance_squared = circle.center().distance_squared(point);
    //             // diameter = 2*radius
    //             // diameter^2 = (2*radius)^2 = 4*(radius^2)
    //             // (diameter^2)/4 = radius^2
    //             let r_squared = (circle.diameter * circle.diameter) as i32 / 4;
    //             center_distance_squared - r_squared
    //         },
    //         SpritePrimitive::Rectangle(rectangle) => {
    //             let p1 = rectangle.closest_edge(point).closest_point(point);
    //             p1.distance_squared(point)
    //         },
    //         SpritePrimitive::Polyline(polyline) => {
    //             let p1 = polyline.closest_edge(point).closest_point(point);
    //             p1.distance_squared(point)
    //         },
    //         SpritePrimitive::Triangle(triangle) => {
    //             let p1 = triangle.closest_edge(point).closest_point(point);
    //             p1.distance_squared(point)
    //         },
    //     }
    // }

    /// approximate distance
    pub fn distance(&self, rhs: &Self) -> i32 {
        let p1 = self.perimeter_point_near(rhs.center());
        let p2 = rhs.perimeter_point_near(self.center());
        p1.distance(p2)
    }

    /// closest point on perimeter to arbitrary point
    pub fn perimeter_point_near(&self, point: Point) -> Point {
                match self {
            SpritePrimitive::Line(line) => line.closest_point(point),
            SpritePrimitive::Circle(circle) => {
                // delta = vector from point to center
                // magnitude 
                // radius = diameter / 2
                // dx = (diam * d.x) / (2 * magnitude)
                // dy = (diam * d.y) / (2 * magnitude)
                let mut delta = point - circle.center();
                let magnitude_x_2 = delta.magnitude() * 2;
                if magnitude_x_2 != 0 {
                    delta.x = (circle.diameter as i32 * delta.x) / magnitude_x_2;
                    delta.y = (circle.diameter as i32 * delta.y) / magnitude_x_2;
                }
                circle.center() + delta
            },
            SpritePrimitive::Rectangle(rectangle) => rectangle.closest_edge(point).closest_point(point),
            SpritePrimitive::Polyline(polyline) => polyline.closest_edge(point).closest_point(point),
            SpritePrimitive::Triangle(triangle) => triangle.closest_edge(point).closest_point(point),
        }
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
    fn surface_normal(&self, point:Point ) -> Point {
        match self {
            SpritePrimitive::Line(line) => line.surface_normal(point),
            SpritePrimitive::Circle(circle) => circle.surface_normal(point),
            SpritePrimitive::Rectangle(rectangle) => rectangle.surface_normal(point),
            SpritePrimitive::Polyline(polyline) => polyline.surface_normal(point),
            SpritePrimitive::Triangle(triangle) => triangle.surface_normal(point),
        }
    }
}

pub type SpriteVelocity = SpriteVectorFloat;

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, defmt::Format)]
pub struct Sprite<'a, C> 
where 
    C: PixelColor,
{
    name: &'a str,
    style: PrimitiveStyle<C>,
    line_style: Option<PrimitiveStyle<C>>,
    shape: SpritePrimitive<'a>,
    area: u32,
    velocity: Option<SpriteVelocity>,
}

impl<'a, C: PixelColor, > Drawable for Sprite<'a, C> {
    type Color = C;

    type Output = ();

    fn draw<D>(&self, target: &mut D) -> Result<Self::Output, D::Error>
    where
        D: DrawTarget<Color = Self::Color> {
        self.shape.draw_styled(&self.style, target)?;
        if let Some(line_style) = self.line_style {
            if let Some(velocity) = self.velocity {
                let center: Point = self.shape.center();
                let delta: Point = velocity.into();
                Line::with_delta(center, delta).draw_styled(&line_style, target)?;
            }
        }
        Ok(())
    }
}

impl<'a, C: PixelColor, > Default for Sprite<'a, C> {
    fn default() -> Self {
        let shape = SpritePrimitive::default();
        let area = shape.area();
        Self { 
            name: Default::default(), 
            style: Default::default(), 
            line_style: Default::default(), 
            shape, 
            area, 
            velocity: Default::default(),
        }
    }
}

impl<'a, C: PixelColor, > Sprite<'a, C> {
    
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

    pub fn with_velocity(self, velocity: SpriteVelocity) -> Self {
        Self {
            velocity: Some(velocity),
            ..self
        }
    }
        
    pub fn shape(&self) -> SpritePrimitive<'a> {
        self.shape
    }
    
    pub fn velocity(&self) -> SpriteVelocity {
        match self.velocity {
            Some(v) => v,
            None => VVelocity::zero(),
        }
    }
    
    pub fn velocity_mut(&mut self) -> &mut Option<SpriteVelocity> {
        &mut self.velocity
    }
    
    pub fn set_velocity(&mut self, velocity: SpriteVelocity) {
        if velocity == VVelocity::zero() {
            self.velocity = None;
        } else {
            self.velocity = Some(velocity);
        }
    }
    
}


impl<'a, C: PixelColor, > Sprite<'a, C> {
    pub fn direction(&self) -> SpriteVelocity {
        match self.velocity {
            Some(v) => v.direction(),
            None => SpriteVelocity::default(),
        }
    }

    pub fn speed(&self) -> f32 {
        match self.velocity {
            Some(v) => v.speed(),
            None => 0.0,
        }        
    }

    pub fn next_position(&self) -> SpritePrimitive<'_> {
        match self.velocity {
            Some(v) => self.shape.translate(v.into()),
            None => self.shape(),
        }
    }

    pub fn area(&self) -> u32 {
        self.area
    }
    
    pub fn is_moving(&self) -> bool {
        self.velocity.is_some()
    }

    pub fn distance_between(&self, other: &Self) -> i32 {
        self.shape.distance(&other.shape)
    }

    #[inline]
    pub fn center(&self) -> Point {
        self.shape.center()
    }
    
    /// rough estimate of distance between the bounding boxes of this and other
    pub fn box_distance(&self, other: &Self) -> i32 {
        let area_of_distance = self.center().distance_squared(other.center());
        let area_of_objects = self.area() + other.area();
        area_of_distance - area_of_objects as i32
    }


    pub fn about_to_collide(&self, other: &Self) -> bool {
        self.box_distance(other) <= 1
        // let current_distance = self.distance_between(other);
        // let p1 = self.next_position();
        // let p2 = other.next_position();
        // // let next_distance = p1.distance(&p2);
        // // if next_distance < current_distance {
        // //     next_distance <= 2
        // // } else {
        // //     false
        // // }
        // current_distance <= 2
    }
    
    pub fn is_collision(&self, sprite2: &Self) -> bool {
        let distance = self.distance_between(sprite2);
        distance < 2
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
        if let Some(mut delta) = self.velocity {
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
        if let Some(delta) = self.velocity {        
            self.shape.translate_mut(delta.into());
        }
    }

    fn calculate_reflection_vector_i32s(velocity: &SpriteVelocity, collision_normal: &SpriteVelocity) -> VelocityCalculationResults<i32> {
        let mut calculations: VelocityCalculationResults<i32> = VelocityCalculationResults::<i32>::default();
        calculations.initial = (*velocity).into();
        calculations.collision_normal = (*collision_normal).into();
        const SCALE: i32 = 1 << 10;
        // let speed = velocity.magnitude();

        // Ensure the normal is normalized (unit length)
        calculations.normal = {
            let mag = collision_normal.magnitude();
            let mut x = calculations.collision_normal.x * SCALE;
            let mut y = calculations.collision_normal.y * SCALE;
            x = (x as f32 / mag).round() as i32;
            y = (y as f32 / mag).round() as i32;
            (x,y).into()
        };
        // Calculate the component of the incoming velocity perpendicular to the collision surface
        calculations.perpendicular = {
            // velocity.perpendicular_velocity(&normal);
            // let normal = collision_normal_normalized;
            let dot = {
                // (v.x * n.x * SCALE) + (v.y * n.x * SCALE) so we divide by scale to eliminate extra scaling
                calculations.initial.dot(*calculations.normal) / SCALE
            };
            calculations.normal * (dot as f32)
        };
        // Calculate the component of the incoming velocity parallel to the collision surface
        calculations.parallel = (calculations.initial * (SCALE)) - calculations.perpendicular;
        // The reflected perpendicular velocity is reversed and scaled by the COR
        calculations.reflected = calculations.perpendicular.neg();

        // The reflected velocity is the sum of the reflected perpendicular and parallel components
        calculations.reflected = calculations.reflected + calculations.parallel;

        calculations.reflected.x /= SCALE;
        calculations.reflected.y /= SCALE;
        calculations
    }

    fn calculate_reflection_vector(velocity: &SpriteVelocity, collision_normal: &SpriteVelocity) -> SpriteVelocity {
        let calc1: VelocityCalculationResults<i32> = Self::calculate_reflection_vector_i32s(velocity, collision_normal);
        info!("calc1: {}", calc1);
        let mut calc2: VelocityCalculationResults<f32> = VelocityCalculationResults::<f32>::default();
        calc2.initial = (*velocity).into();
        calc2.collision_normal = (*collision_normal).into();

        // let speed = velocity.magnitude();
        // Ensure the normal is normalized (unit length)
        calc2.normal = {
            let mag = collision_normal.magnitude();
            let mut x = calc2.collision_normal.x;
            let mut y = calc2.collision_normal.y;
            x = (x / mag);
            y = (y / mag);
            (x,y).into()
        };

        // Calculate the component of the incoming velocity perpendicular to the collision surface
        calc2.perpendicular = {
            let dot = {
                // (v.x * n.x * SCALE) + (v.y * n.x * SCALE) so we divide by scale to eliminate extra scaling
                calc2.initial.dot(*calc2.normal)
            };
            calc2.normal * (dot as f32)
        };

        // Calculate the component of the incoming velocity parallel to the collision surface
        calc2.parallel =  calc2.initial - calc2.perpendicular;

        // The reflected perpendicular velocity is reversed and scaled by the COR
        calc2.reflected = -calc2.perpendicular;

        // The reflected velocity is the sum of the reflected perpendicular and parallel components
        calc2.reflected = calc2.reflected + calc2.parallel;

        info!("calc2: {}", calc2);
        // let speed2 = reflection.magnitude();
        // if (speed - speed2).abs() >= 3.0*f32::EPSILON {
        //     info!("speed changed by {}", (speed - speed2));
        // }
        // let ref_norm = reflection.normalize();
        // ref_norm * Coefficient(speed)
        calc2.reflected.into()
    }

    pub fn update_velocity(&mut self, other: &Self) -> bool {        
        let vec = other.shape.surface_normal(self.shape.center());
        if vec == Point::zero() {
            panic!("surface normal for {} is zero", other.name);
        }

        let collision_normal = SpriteVelocity::from(vec);
        let v1 = self.velocity();
        let v2 = Self::calculate_reflection_vector(&v1, &collision_normal);
        // if v1 == v2 {
        //     info!("collision_normal: {}, v1: {}, v2: {}", collision_normal, v1, v2);
        // }
        // info!("sprite: {}, v1: {}, v2: {}", self.name, v1, v2);
        // let speed_a = vel.magnitude();
        // let speed_b = velocity.magnitude();
        // if (speed_a - speed_b) > 0.1 {
        //     info!("sprite: {}, collision_normal: {}", other.name, collision_normal);
        //     info!("sprite: {}, vel1: {}, vel2: {}", self.name, vel, velocity);
        //     info!("sprite: {} slowed down, speed before: {}, after: {}", self.name, speed_a, speed_b);
        // }
        self.set_velocity(v2);
        v1 != v2
    }

}


// #[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
#[derive(Clone, Debug, Default, defmt::Format)]
pub struct SpriteContainer<'a, C: PixelColor> 
    where 
        C: PixelColor, 
{
    boundary: Rectangle,
    sprites: Vec<Sprite<'a, C>, 10>,
    _phantom: core::marker::PhantomData<C>,
}

impl<'a, C: PixelColor> SpriteContainer<'a, C> {
    pub fn new(boundary: Rectangle) -> Self {
        let sprites = Vec::new();
        Self { boundary, sprites, _phantom: core::marker::PhantomData }
    }

    pub fn add_sprite(&mut self, sprite: Sprite<'a, C>) -> Result<(), Sprite<'a, C>> {
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
                                info!("{} bounced off of {}", current_sprite.name, other_sprite.name);
                            } else {
                                info!("{} and {} hit with no bounce", current_sprite.name, other_sprite.name);
                            }
                        }
                        // if current_sprite.about_to_collide(other_sprite) 
                        //     && current_sprite.update_velocity(other_sprite) {
                        //         // info!("{} bounced off of {}", current_sprite.name(), other_sprite.name());                            
                        // }
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

impl<'a, C: PixelColor> Drawable for SpriteContainer<'a, C> {
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

