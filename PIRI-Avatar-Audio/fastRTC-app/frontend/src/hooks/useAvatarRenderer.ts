/**
 * Custom hook for avatar rendering using Three.js and VRM
 */

import { useEffect, useRef, useState } from "react";
import { BlendShapeValues } from "@/src/types/avatar.types";

interface UseAvatarRendererOptions {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  modelPath?: string;
}

export function useAvatarRenderer({
  canvasRef,
  modelPath = "/models/avatar.vrm",
}: UseAvatarRendererOptions) {
  const sceneRef = useRef<any>(null);
  const rendererRef = useRef<any>(null);
  const vrmRef = useRef<any>(null);
  const animationIdRef = useRef<number | null>(null);
  const [vrmLoaded, setVrmLoaded] = useState(false);

  useEffect(() => {
    if (!canvasRef.current) return;

    let mounted = true;

    const initializeRenderer = async () => {
      try {
        // Dynamic imports for Three.js
        const THREE = await import("three");
        const { GLTFLoader } = await import(
          "three/addons/loaders/GLTFLoader.js"
        );

        // Try to import VRM support
        let VRMLoaderPlugin: any = null;
        let VRMUtils: any = null;
        let isVRMSupported = false;

        try {
          const vrmModule = await import("@pixiv/three-vrm");
          VRMLoaderPlugin = vrmModule.VRMLoaderPlugin;
          VRMUtils = vrmModule.VRMUtils;
          isVRMSupported = true;
        } catch (vrmError) {
          console.warn(
            "VRM support not available, falling back to GLB:",
            vrmError
          );
          isVRMSupported = false;
        }

        if (!mounted || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const container = canvas.parentElement;
        if (!container) return;

        const aspect = container.clientWidth / container.clientHeight || 1;

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        // Camera setup (optimized for face focus)
        const camera = new THREE.PerspectiveCamera(35, aspect, 0.1, 1000);
        camera.position.set(0, 1.4, 3.5); // Moved back for better framing
        camera.lookAt(0, 1.4, 0);

        // Renderer setup
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

        // Enhanced lighting for VRM models
        const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
        scene.add(hemi);

        const key = new THREE.DirectionalLight(0xffffff, 1.2);
        key.position.set(0, 1, 2);
        scene.add(key);

        const rim = new THREE.DirectionalLight(0xffffff, 0.6);
        rim.position.set(0, 1, -2);
        scene.add(rim);

        // Initialize GLTF loader
        const gltfLoader = new GLTFLoader();

        // Add VRM plugin if available
        if (isVRMSupported && VRMLoaderPlugin) {
          gltfLoader.register((parser) => new VRMLoaderPlugin(parser));
        }

        // Store references
        sceneRef.current = scene;
        rendererRef.current = renderer;

        const avatarPaths = [modelPath];

        for (const path of avatarPaths) {
          try {
            console.log(`Loading avatar: ${path}`);

            const gltf = await new Promise<any>((resolve, reject) => {
              gltfLoader.load(path, resolve, undefined, reject);
            });

            // Check if this is a VRM file
            const vrm = gltf.userData?.vrm;
            const isVRM = !!vrm;

            if (vrmRef.current) {
              scene.remove(vrmRef.current.scene || vrmRef.current);
            }

            if (isVRM) {
              // Handle VRM
              vrmRef.current = vrm;
              scene.add(vrm.scene);

              // VRM-specific optimizations
              if (VRMUtils) {
                VRMUtils.removeUnnecessaryVertices(vrm.scene);
                VRMUtils.removeUnnecessaryJoints(vrm.scene);
              }

              // Log VRM blend shapes for debugging
              if (vrm.expressionManager) {
                console.log(
                  "VRM expressions:",
                  Object.keys(vrm.expressionManager.expressionMap || {})
                );
              }

              // Enhanced VRM blend shape detection
              vrm.scene.traverse((child: any) => {
                if (child.isMesh && child.morphTargetInfluences) {
                  child.userData.morphTargets = child.morphTargetDictionary;
                  if (child.morphTargetDictionary) {
                    const morphNames = Object.keys(child.morphTargetDictionary);
                    console.log("🎭 VRM morph targets found:", morphNames);
                  }
                }
              });

              const box = new THREE.Box3().setFromObject(vrm.scene);
              const center = box.getCenter(new THREE.Vector3());
              const size = box.getSize(new THREE.Vector3());

              const faceTarget = center.clone();
              faceTarget.y += size.y * 0.35;

              camera.position.set(
                0,
                1.5,
                2.5 // Adjusted for better face framing
              );
              camera.lookAt(faceTarget);

              console.log(`✅ VRM avatar loaded: ${path}`);
            } else if (gltf.scene) {
              // Handle GLB if a scene exists
              vrmRef.current = gltf.scene;
              scene.add(gltf.scene);

              // Keep original camera position
              // Camera is already positioned at (0, 1.5, 2) looking at (0, 1.5, 0)

              // Find morph targets for GLB
              gltf.scene.traverse((child: any) => {
                if (child.isMesh && child.morphTargetInfluences) {
                  child.userData.morphTargets = child.morphTargetDictionary;
                  if (child.morphTargetDictionary) {
                    console.log(
                      "🎭 GLB morph targets found:",
                      Object.keys(child.morphTargetDictionary)
                    );
                  }
                }
              });

              console.log(`✅ GLB avatar loaded: ${path}`);
            }

            setVrmLoaded(true);
            break;
          } catch (error) {
            console.log(`Failed to load: ${error}`);
          }
        }

        if (!vrmRef.current) {
          console.log(
            "❌ No avatar found. Please place a .vrm or .glb file in /static/"
          );
          setVrmLoaded(false);
        }

        // Store references
        sceneRef.current = scene;
        rendererRef.current = renderer;

        // Start render loop
        const animate = () => {
          requestAnimationFrame(animate);

          // Update VRM if available
          if (vrmRef.current && vrmRef.current.update) {
            vrmRef.current.update(0.016); // ~60fps delta time
          }

          renderer.render(scene, camera);
        };
        animate();

        // Handle window resize
        const handleResize = () => {
          if (!container || !sceneRef.current) return;
          const { camera, renderer } = sceneRef.current;
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(container.clientWidth, container.clientHeight);
        };

        window.addEventListener("resize", handleResize);

        return () => {
          mounted = false;
          window.removeEventListener("resize", handleResize);

          if (animationIdRef.current) {
            cancelAnimationFrame(animationIdRef.current);
          }

          if (rendererRef.current) {
            rendererRef.current.dispose();
          }

          if (sceneRef.current) {
            sceneRef.current.scene.clear();
          }
        };
      } catch (error) {
        console.error("Failed to initialize avatar renderer:", error);
      }
    };

    initializeRenderer();
  }, [canvasRef, modelPath]);

  const applyBlendShapes = (blendShapes: BlendShapeValues) => {
    if (!vrmRef.current) return;

    // Handle VRM 1.0 expressions
    if (vrmRef.current.expressionManager) {
      Object.entries(blendShapes).forEach(([name, value]) => {
        try {
          // VRM 1.0 uses expressionManager
          vrmRef.current.expressionManager.setValue(name, value);
        } catch (error) {
          console.warn(`Failed to apply expression ${name}:`, error);
        }
      });
      vrmRef.current.expressionManager.update();
    }
    // Handle VRM 0.0 blend shapes
    else if (vrmRef.current.blendShapeProxy) {
      Object.entries(blendShapes).forEach(([name, value]) => {
        try {
          vrmRef.current.blendShapeProxy.setValue(name, value);
        } catch (error) {
          console.warn(`Failed to apply blend shape ${name}:`, error);
        }
      });
      vrmRef.current.blendShapeProxy.update();
    }
  };

  return {
    applyBlendShapes,
    vrm: vrmRef.current,
    vrmLoaded,
  };
}
