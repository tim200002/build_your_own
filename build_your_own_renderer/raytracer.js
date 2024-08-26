const WIDTH = 256;
const HEIGHT = 192;

const image = new Image(WIDTH, HEIGHT);

class RayTracer {
  constructor(scene, w, h) {
    this.scene = scene;
    this.w = w;
    this.h = h;

    this.bgColor = new Color(0.0, 0, 0);
    this.max_recursion_depth = 3;
  }

  tracedValueAtPixel(x, y) {
    const alpha = x / this.w;
    const beta = y / this.h;

    const t = Vector3D.lerp(
      this.scene.imagePlane.topLeft,
      this.scene.imagePlane.topRight,
      alpha
    );
    const b = Vector3D.lerp(
      this.scene.imagePlane.bottomLeft,
      this.scene.imagePlane.bottomRight,
      alpha
    );

    const point = Vector3D.lerp(t, b, beta);
    const ray = new Ray(point, point.subtract(this.scene.camera));

    return this.calc_ray_color(ray, 0);
  }

  calc_ray_color(ray, recursion_depth) {
    let sphere_t = null;
    let sphere_intersect = null;
    for (const sphere of this.scene.spheres) {
      const [is_intersecting, t] = sphere.checkCollisionWithRay(ray);
      if (is_intersecting) {
        if (!sphere_t || t < sphere_t) {
          sphere_t = t;
          sphere_intersect = sphere;
        }
      }
    }

    if (!sphere_intersect) {
      return this.bgColor;
    }

    const pointOfIntersection = ray.origin.add(ray.direction.scale(sphere_t));
    const surfaceNormal = pointOfIntersection
      .subtract(sphere_intersect.center)
      .normalize();

    // Calc ambiance color
    const ambianceColor = this.scene.ambientLight.scale(
      sphere_intersect.material.k_a
    );

    // cast a ray to all light sources an check whether we collide with any other object
    const nonOccludedLights = [];
    for (const light of this.scene.lights) {
      const ray = new Ray(
        pointOfIntersection,
        light.origin.subtract(pointOfIntersection)
      );
      let isOccluded = false;
      for (const sphere of this.scene.spheres) {
        if (sphere === sphere_intersect) {
          continue;
        }
        const [is_intersecting, t] = sphere.checkCollisionWithRay(ray);
        if (is_intersecting && t > 0 && t <= 1) {
          isOccluded = true;
          break;
        }
      }
      if (!isOccluded) {
        nonOccludedLights.push(light);
      }
    }

    if (nonOccludedLights.length == 0) {
      return ambianceColor;
    }

    // Calc diffusion color
    let diffuseColor = new Color(0, 0, 0);
    for (const light of nonOccludedLights) {
      const lightVector = light.origin
        .subtract(pointOfIntersection)
        .normalize();

      const dp = surfaceNormal.dot(lightVector);
      if (dp < 0) {
        continue; // ignore lights shining form the inside out
      }
      const diffuseColorLight = light.i_d
        .scale(sphere_intersect.material.k_d)
        .scale(dp);
      diffuseColor = diffuseColor.add(diffuseColorLight);
    }

    // calc specular color
    let specularColor = new Color(0, 0, 0);
    for (const light of nonOccludedLights) {
      const lightVector = light.origin
        .subtract(pointOfIntersection)
        .normalize();
      const dp = surfaceNormal.dot(lightVector);

      const reflectance = surfaceNormal.scale(2 * dp).subtract(lightVector);
      const viewVector = this.scene.camera
        .subtract(pointOfIntersection)
        .normalize();
      const specTerm = Math.pow(
        reflectance.dot(viewVector),
        sphere_intersect.material.alpha
      );

      const specularColorLocal = light.i_s
        .scale(sphere_intersect.material.k_s)
        .scale(specTerm);

      specularColor = specularColor.add(specularColorLocal);
    }

    let color = ambianceColor.add(diffuseColor).add(specularColor).clamp();

    if (recursion_depth < this.max_recursion_depth) {
      const V_hat = ray.direction.scale(-1).normalize();
      const reflectanceVec = surfaceNormal
        .scale(2 * V_hat.dot(surfaceNormal))
        .subtract(V_hat);
      const reflectance_ray = new Ray(pointOfIntersection, reflectanceVec);
      const reflected_color = this.calc_ray_color(
        reflectance_ray,
        recursion_depth + 1
      );
      color = color.add(reflected_color.scale(sphere_intersect.material.k_r));
    }

    return color;
  }
}

const convertColor = (color) => {
  return {
    r: Math.floor(color.r * 255),
    g: Math.floor(color.g * 255),
    b: Math.floor(color.b * 255)
  };
};

const Scene = {
  camera: new Vector3D(0, 0, 2),

  imagePlane: {
    topLeft: new Vector3D(-1.28, 0.86, 0),
    topRight: new Vector3D(1.28, 0.86, 0),
    bottomLeft: new Vector3D(-1.28, -0.86, 0),
    bottomRight: new Vector3D(1.28, -0.86, 0)
  },

  spheres: [
    new Sphere(
      new Vector3D(-1.1, 0.6, -1),
      0.4,
      new Material(
        new Color(0.1, 0.1, 0.1),
        new Color(0.5, 0.5, 0.9),
        new Color(0.7, 0.7, 0.7),
        20,
        new Color(0.1, 0.1, 0.2)
      )
    ),
    //new Sphere(new Vector3D(-1, 1, 2), 1, new Material(0.3, 0.5, 0.8, 0.4)),
    new Sphere(
      new Vector3D(0.2, -0.1, -1),
      0.5,
      new Material(
        new Color(0.1, 0.1, 0.1),
        new Color(0.9, 0.5, 0.5),
        new Color(0.7, 0.7, 0.7),
        20,
        new Color(0.2, 0.1, 0.1)
      )
    ),
    new Sphere(
      new Vector3D(1.2, -0.5, -1.75),
      0.4,
      new Material(
        new Color(0.1, 0.1, 0.1),
        new Color(0.5, 0.9, 0.5),
        new Color(0.7, 0.7, 0.7),
        20,
        new Color(0.8, 0.9, 0.8)
      )
    )
  ],
  ambientLight: new Color(0.5, 0.5, 0.5),
  lights: [
    new Light(
      new Vector3D(-3, -0.5, 1),
      new Color(0.8, 0.3, 0.3),
      new Color(0.8, 0.8, 0.8)
    ),
    new Light(
      new Vector3D(3, 2, 1),
      new Color(0.4, 0.4, 0.9),
      new Color(0.8, 0.8, 0.8)
    )
  ]
};

document.image = image;
const rayTracer = new RayTracer(Scene, WIDTH, HEIGHT);

for (let y = 0; y < HEIGHT; y++) {
  for (let x = 0; x < WIDTH; x++) {
    color = rayTracer.tracedValueAtPixel(x, y);

    image.putPixel(x, y, convertColor(color));
  }
}

image.renderInto(document.querySelector('body'));
