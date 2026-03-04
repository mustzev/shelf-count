import { useCallback, useRef } from 'react'
import {
  type Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  VisionCameraProxy,
} from 'react-native-vision-camera'
import { useSharedValue, Worklets } from 'react-native-worklets-core'
import type { TensorflowModel } from 'react-native-fast-tflite'
import { processYoloOutputs } from '../ml/postprocess'
import type { DetectionResult } from '../ml/types'
import { setResult } from '../frameStore'

const CONFIDENCE_THRESHOLD = 0.25
const IOU_THRESHOLD = 0.5
const MODEL_INPUT_SIZE = 640

const resizePlugin = VisionCameraProxy.initFrameProcessorPlugin('resize', {})

type CaptureResolve = (result: DetectionResult) => void

export function useCamera(model: TensorflowModel | undefined) {
  const cameraRef = useRef<Camera>(null)
  const device = useCameraDevice('back')
  const { hasPermission, requestPermission } = useCameraPermission()
  const captureRequested = useSharedValue(false)
  const resolveRef = useRef<CaptureResolve | null>(null)
  // Ref so the bridge callback always sees the latest model.
  const modelRef = useRef<TensorflowModel | undefined>(model)
  modelRef.current = model

  const onPixelsReady = Worklets.createRunOnJS((pixels: number[], frameWidth: number, frameHeight: number) => {
    try {
      if (!modelRef.current) {
        console.error('[useCamera] model not loaded')
        return
      }

      // YOLOv8 expects float32 input normalized to 0–1.
      // pixels[] contains uint8 RGB values from the resize plugin.
      const float32Input = new Float32Array(pixels.length)
      for (let i = 0; i < pixels.length; i++) {
        float32Input[i] = pixels[i] / 255
      }

      // Run inference on the JS thread.
      const start = performance.now()
      const outputs = modelRef.current.runSync([float32Input])
      const timeMs = performance.now() - start

      // YOLOv8 output: single tensor [1, 5, 8400] — flat Float32Array.
      const raw = outputs[0] as Float32Array
      let detections = processYoloOutputs(raw, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

      // Camera sensor is landscape on Android. When phone is portrait,
      // frame coordinates are rotated 90° from the displayed photo.
      // Rotate bbox coordinates to match the photo orientation.
      if (frameWidth > frameHeight) {
        detections = detections.map((d) => ({
          ...d,
          bbox: {
            x: d.bbox.y,
            y: 1 - d.bbox.x - d.bbox.width,
            width: d.bbox.height,
            height: d.bbox.width,
          },
        }))
      }

      console.log(
        `[useCamera] detections: ${detections.length} (frame: ${frameWidth}x${frameHeight})`,
        detections.slice(0, 3).map((d) =>
          `${d.label}(${d.confidence.toFixed(2)}) @${d.bbox.x.toFixed(2)},${d.bbox.y.toFixed(2)} ${d.bbox.width.toFixed(3)}x${d.bbox.height.toFixed(3)}`
        ).join(' | '),
      )

      const result: DetectionResult = {
        detections,
        count: detections.length,
        inferenceTimeMs: timeMs,
      }
      setResult(result)
      if (resolveRef.current) {
        resolveRef.current(result)
        resolveRef.current = null
      }
    } catch (e) {
      console.error('[useCamera] JS inference error:', e)
      const result: DetectionResult = {
        detections: [],
        count: 0,
        inferenceTimeMs: 0,
      }
      setResult(result)
      if (resolveRef.current) {
        resolveRef.current(result)
        resolveRef.current = null
      }
    }
  })

  const onFrameError = Worklets.createRunOnJS((msg: string) => {
    console.error('[useCamera]', msg)
    const result: DetectionResult = {
      detections: [],
      count: 0,
      inferenceTimeMs: 0,
    }
    setResult(result)
    if (resolveRef.current) {
      resolveRef.current(result)
      resolveRef.current = null
    }
  })

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      if (captureRequested.value && model != null && resizePlugin != null) {
        captureRequested.value = false

        try {
          const rawBuffer = resizePlugin.call(frame, {
            scale: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE },
            pixelFormat: 'rgb',
            dataType: 'uint8',
          })

          // Read bytes via DataView → plain number[] that can cross the bridge.
          const dv = new DataView(rawBuffer as unknown as ArrayBuffer)
          const size = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3
          const pixels: number[] = new Array(size)
          for (let i = 0; i < size; i++) {
            pixels[i] = dv.getUint8(i)
          }

          // Send pixels + frame dimensions to JS thread for float conversion + inference.
          onPixelsReady(pixels, frame.width, frame.height)
        } catch (e) {
          onFrameError('Worklet error: ' + String(e))
        }
      }
    },
    [captureRequested, model, onPixelsReady, onFrameError],
  )

  const captureFrame = useCallback((): Promise<DetectionResult> => {
    return new Promise((resolve) => {
      resolveRef.current = resolve
      captureRequested.value = true
    })
  }, [captureRequested])

  const takePhoto = useCallback(async () => {
    if (!cameraRef.current) return null
    return cameraRef.current.takePhoto({ enableShutterSound: false })
  }, [])

  return {
    cameraRef,
    device,
    hasPermission,
    requestPermission,
    takePhoto,
    captureFrame,
    frameProcessor,
    modelReady: model != null,
  }
}
