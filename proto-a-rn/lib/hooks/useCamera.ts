import { useCallback, useRef } from 'react'
import {
  type Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera'
import { useSharedValue, Worklets } from 'react-native-worklets-core'
import { type FrameData, setFrame } from '../frameStore'

type CaptureResolve = (data: FrameData) => void

export function useCamera() {
  const cameraRef = useRef<Camera>(null)
  const device = useCameraDevice('back')
  const { hasPermission, requestPermission } = useCameraPermission()
  const captureRequested = useSharedValue(false)
  const resolveRef = useRef<CaptureResolve | null>(null)

  const onFrameCaptured = Worklets.createRunOnJS(
    (
      buffer: ArrayBuffer,
      width: number,
      height: number,
      bytesPerRow: number,
    ) => {
      const data: FrameData = {
        pixels: new Uint8Array(buffer),
        width,
        height,
        bytesPerRow,
      }
      setFrame(data)
      if (resolveRef.current) {
        resolveRef.current(data)
        resolveRef.current = null
      }
    },
  )

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      if (captureRequested.value) {
        captureRequested.value = false
        const buffer = frame.toArrayBuffer()
        onFrameCaptured(buffer, frame.width, frame.height, frame.bytesPerRow)
      }
    },
    [captureRequested, onFrameCaptured],
  )

  const captureFrame = useCallback((): Promise<FrameData> => {
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
  }
}
