import { useRef, useEffect } from 'react';
import { Renderer, Program, Mesh, Triangle, Vec2 } from 'ogl';

const vertex = `
attribute vec2 position;
attribute vec2 uv;
void main(){
    gl_Position=vec4(position,0.0,1.0);
}
`;

const fragment = `
#ifdef GL_ES
precision highp float;
#endif
uniform vec2 uResolution;
uniform float uTime;

float noise(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float smoothNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = noise(i);
    float b = noise(i + vec2(1.0, 0.0));
    float c = noise(i + vec2(0.0, 1.0));
    float d = noise(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for(int i = 0; i < 6; i++) {
        value += amplitude * smoothNoise(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

void main() {
    vec2 uv = gl_FragCoord.xy / uResolution.xy;
    vec2 p = uv * 3.0;
    
    float t = uTime * 0.3;
    
    // Create flowing patterns
    vec2 q = vec2(fbm(p + t * 0.1), fbm(p + vec2(1.0)));
    vec2 r = vec2(fbm(p + q + t * 0.15), fbm(p + q + vec2(1.7, 9.2)));
    
    float f = fbm(p + r);
    
    // PURE WHITE background with DARK GREEN patterns
    vec3 darkGreen = vec3(0.15, 0.55, 0.30);    // Dark visible green
    vec3 pureWhite = vec3(1.0, 1.0, 1.0);       // Pure white
    
    // Create strong contrast
    float patternStrength = smoothstep(0.42, 0.58, f);
    vec3 color = mix(darkGreen, pureWhite, patternStrength);
    
    // Add detail
    float detailPattern = smoothstep(0.45, 0.55, r.x);
    vec3 mediumGreen = vec3(0.20, 0.60, 0.35);
    color = mix(color, mediumGreen, (1.0 - detailPattern) * (1.0 - patternStrength) * 0.4);
    
    gl_FragColor = vec4(color, 1.0);
}
`;

export default function DarkVeil2() {
  const ref = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = ref.current as HTMLCanvasElement;
    const parent = canvas.parentElement as HTMLElement;

    const renderer = new Renderer({
      dpr: Math.min(window.devicePixelRatio, 2),
      canvas
    });

    const gl = renderer.gl;
    const geometry = new Triangle(gl);

    const program = new Program(gl, {
      vertex,
      fragment,
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new Vec2() }
      }
    });

    const mesh = new Mesh(gl, { geometry, program });

    const resize = () => {
      const w = parent.clientWidth;
      const h = parent.clientHeight;
      renderer.setSize(w, h);
      program.uniforms.uResolution.value.set(w, h);
    };

    window.addEventListener('resize', resize);
    resize();

    const start = performance.now();
    let frame = 0;

    const loop = () => {
      program.uniforms.uTime.value = (performance.now() - start) / 1000;
      renderer.render({ scene: mesh });
      frame = requestAnimationFrame(loop);
    };

    loop();

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
    };
  }, []);
  
  return <canvas ref={ref} className="w-full h-full block" />;
}
