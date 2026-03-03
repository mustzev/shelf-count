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
import { processPostNMSOutputs } from '../ml/postprocess'
import type { DetectionResult } from '../ml/types'
import { setResult } from '../frameStore'

const CONFIDENCE_THRESHOLD = 0.4

const resizePlugin = VisionCameraProxy.initFrameProcessorPlugin('resize', {})

type CaptureResolve = (result: DetectionResult) => void

export function useCamera(
  model: TensorflowModel | undefined,
  labels: (string | null)[],
) {
  const cameraRef = useRef<Camera>(null)
  const device = useCameraDevice('back')
  const { hasPermission, requestPermission } = useCameraPermission()
  const captureRequested = useSharedValue(false)
  const resolveRef = useRef<CaptureResolve | null>(null)
  // Ref so the bridge callback always sees the latest model/labels.
  const modelRef = useRef<TensorflowModel | undefined>(model)
  modelRef.current = model
  const labelsRef = useRef(labels)
  labelsRef.current = labels

  const onPixelsReady = Worklets.createRunOnJS((pixels: number[]) => {
    try {
      if (!modelRef.current) {
        console.error('[useCamera] model not loaded')
        return
      }

      // Model expects uint8 — pass Uint8Array directly.
      const resized = new Uint8Array(pixels)

      // Run inference on the JS thread.
      const start = performance.now()
      const outputs = modelRef.current.runSync([resized])
      const timeMs = performance.now() - start

      // Post-NMS EfficientDet-Lite0 outputs (4 tensors):
      //   [0] boxes  [1, 25, 4] — normalized [y1, x1, y2, x2]
      //   [1] classes [1, 25]   — class indices (float)
      //   [2] scores  [1, 25]   — confidence [0, 1]
      //   [3] count   [1]       — number of valid detections
      const detections = processPostNMSOutputs(
        outputs[0] as Float32Array,
        outputs[1] as Float32Array,
        outputs[2] as Float32Array,
        outputs[3] as Float32Array,
        labelsRef.current,
        CONFIDENCE_THRESHOLD,
      )
      console.log(
        `[useCamera] detections: ${detections.length}`,
        detections.map((d) => `${d.label}(${d.confidence.toFixed(2)})`).join(', '),
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
          // Resize frame to 320x320 RGB in native C++.
          const rawBuffer = resizePlugin.call(frame, {
            scale: { width: 320, height: 320 },
            pixelFormat: 'rgb',
            dataType: 'uint8',
          })

          // Read bytes via DataView → plain number[] that can cross the bridge.
          const dv = new DataView(rawBuffer as unknown as ArrayBuffer)
          const size = 320 * 320 * 3
          const pixels: number[] = new Array(size)
          for (let i = 0; i < size; i++) {
            pixels[i] = dv.getUint8(i)
          }

          // Send pixels to JS thread for inference.
          onPixelsReady(pixels)
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
