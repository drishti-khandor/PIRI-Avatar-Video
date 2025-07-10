/**
 * Custom hook for avatar rendering using Three.js and VRM
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { BlendShapeValues } from "@/src/types/avatar.types";

interface UseAvatarRendererOptions {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  modelPath?: string;
}

export function useAvatarRenderer({
  canvasRef,
  modelPath = "/models/avatar.vrm",
}: UseAvatarRendererOptions) {
  const sceneRef = useRef<any>(null);
  const rendererRef = useRef<any>(null);
  const vrmRef = useRef<any>(null);
  const blendShapeMapRef = useRef<Record<string, number>>({});
  const animationIdRef = useRef<number | null>(null);
  const threeRef = useRef<any>(null);
  const [vrmLoaded, setVrmLoaded] = useState(false);

  // Unified applyBlendShapes using proxy or direct morph influences
  const applyBlendShapes = useCallback((blendShapes: BlendShapeValues) => {
    const vrm = vrmRef.current;
    const map = blendShapeMapRef.current;
    const THREE = threeRef.current;
    if (!vrm || !map || !THREE) return;

    // VRM 1.0 API: blendShapeProxy
    const proxy = vrm.blendShapeProxy;
    if (proxy) {
      Object.entries(blendShapes).forEach(([name, value]) => {
        if (map[name] === undefined) return;
        proxy.setValue(name, THREE.MathUtils.clamp(value, 0, 1));
      });
      proxy.update();
      return;
    }

    // VRM 0.0 API: expressionManager fallback

    const exprMgr = vrm.expressionManager;
    if (exprMgr) {
      Object.entries(blendShapes).forEach(([name, value]) => {
        if (map[name] === undefined) return;
        exprMgr.setValue(name, THREE.MathUtils.clamp(value, 0, 1));
      });
      exprMgr.update();
      return;
    }

    // Direct morphTargetInfluences fallback
    vrm.scene.traverse((child: any) => {
      if (!child.isMesh || !child.morphTargetInfluences) return;
      // optional: smooth decay
      for (let i = 0; i < child.morphTargetInfluences.length; i++) {
        child.morphTargetInfluences[i] *= 0.9;
      }
      Object.entries(blendShapes).forEach(([name, value]) => {
        const idx = map[name];
        if (idx === undefined) return;
        const v = THREE.MathUtils.clamp(value, 0, 1);
        child.morphTargetInfluences[idx] +=
          (v - child.morphTargetInfluences[idx]) * 0.3;
      });
    });
  }, []);

  useEffect(() => {
    if (!canvasRef.current) return;
    let mounted = true;

    const initializeRenderer = async () => {
      try {
        const THREE = await import("three");
        threeRef.current = THREE;
        const { GLTFLoader } = await import(
          "three/addons/loaders/GLTFLoader.js"
        );

        let VRMLoaderPlugin: any = null;
        let VRMUtils: any = null;
        let isVRMSupported = false;

        try {
          const vrmModule = await import("@pixiv/three-vrm");
          VRMLoaderPlugin = vrmModule.VRMLoaderPlugin;
          VRMUtils = vrmModule.VRMUtils;
          isVRMSupported = true;
        } catch {
          isVRMSupported = false;
        }

        if (!mounted || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const container = canvas.parentElement;
        if (!container) return;
        const aspect = container.clientWidth / container.clientHeight || 1;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);
        const camera = new THREE.PerspectiveCamera(35, aspect, 0.1, 1000);
        camera.position.set(0, 1.4, 1);
        camera.lookAt(0, 1.4, 0);

        const renderer = new THREE.WebGLRenderer({
          canvas,
          antialias: true,
          preserveDrawingBuffer: true,
          powerPreference: "high-performance",
        });
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.0;
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;

        // lights
        scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.8));
        const key = new THREE.DirectionalLight(0xffffff, 1.2);
        key.position.set(0, 1, 2);
        scene.add(key);
        const rim = new THREE.DirectionalLight(0xffffff, 0.6);
        rim.position.set(0, 1, -2);
        scene.add(rim);

        const gltfLoader = new GLTFLoader();
        if (isVRMSupported && VRMLoaderPlugin) {
          gltfLoader.register((parser) => new VRMLoaderPlugin(parser));
        }

        sceneRef.current = scene;
        rendererRef.current = renderer;

        // load VRM
        const gltf = await new Promise<any>((resolve, reject) => {
          gltfLoader.load(modelPath, resolve, undefined, reject);
        });
        const vrm = gltf.userData.vrm || null;
        if (!vrm) throw new Error("Loaded file is not a VRM");

        vrmRef.current = vrm;
        scene.add(vrm.scene);
        if (VRMUtils) VRMUtils.removeUnnecessaryJoints(vrm.scene);

        // build blend-shape index map
        const map: Record<string, number> = {};
        // VRM1.0: blendShapeProxy groups
        if (vrm.blendShapeProxy) {
          vrm.blendShapeProxy.blendShapeGroups.forEach((g: any) => {
            map[g.blendShapeName] = g.index;
          });
        }
        // fallback meshes
        vrm.scene.traverse((child: any) => {
          if (child.isMesh && child.morphTargetDictionary) {
            Object.entries(child.morphTargetDictionary).forEach(([n, i]) => {
              map[n] = i as number;
            });
          }
        });
        blendShapeMapRef.current = map;
        console.log("Blend-shape map:", map);

        setVrmLoaded(true);

        // render loop
        const animate = () => {
          animationIdRef.current = requestAnimationFrame(animate);
          vrm.update?.(0.016);
          renderer.render(scene, camera);
        };
        animate();

        window.addEventListener("resize", () => {
          const w = container.clientWidth;
          const h = container.clientHeight;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        });
      } catch (err) {
        console.error("Avatar renderer init error:", err);
      }
    };

    initializeRenderer();
    return () => {
      mounted = false;
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      rendererRef.current?.dispose();
      sceneRef.current?.clear();
    };
  }, [canvasRef, modelPath]);

  return { applyBlendShapes, vrm: vrmRef.current, vrmLoaded };
}
