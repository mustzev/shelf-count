import { useRouter } from 'expo-router'
import { useEffect } from 'react'
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native'
import { Camera } from 'react-native-vision-camera'
import { useTensorflowModel } from 'react-native-fast-tflite'
import { useCamera } from '../../lib/hooks/useCamera'
import { COCO_90_LABELS } from '../../lib/ml/labels'

export default function CameraScreen() {
  const router = useRouter()
  const tfModel = useTensorflowModel(
    require('../../assets/models/efficientdet-lite0-v2.tflite'),
  )
  const model = tfModel.state === 'loaded' ? tfModel.model : undefined
  const {
    cameraRef,
    device,
    hasPermission,
    requestPermission,
    takePhoto,
    captureFrame,
    frameProcessor,
    modelReady,
  } = useCamera(model, COCO_90_LABELS)

  useEffect(() => {
    if (!hasPermission) {
      requestPermission()
    }
  }, [hasPermission, requestPermission])

  const handleCapture = async () => {
    const [_result, photo] = await Promise.all([captureFrame(), takePhoto()])
    if (photo) {
      router.push({
        pathname: '/(tabs)/results',
        params: { photoPath: photo.path },
      })
    }
  }

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>Camera permission required</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    )
  }

  if (!device) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>No camera device found</Text>
      </View>
    )
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
        pixelFormat="rgb"
        frameProcessor={frameProcessor}
      />
      {!modelReady && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="small" color="#fff" />
          <Text style={styles.loadingText}>Loading model…</Text>
        </View>
      )}
      <View style={styles.captureContainer}>
        <TouchableOpacity
          style={[
            styles.captureButton,
            !modelReady && styles.captureButtonDisabled,
          ]}
          onPress={handleCapture}
          disabled={!modelReady}
        >
          <View style={styles.captureInner} />
        </TouchableOpacity>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  message: {
    color: '#fff',
    fontSize: 16,
    marginBottom: 16,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 60,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    gap: 8,
  },
  loadingText: {
    color: '#fff',
    fontSize: 13,
  },
  captureContainer: {
    position: 'absolute',
    bottom: 40,
    alignSelf: 'center',
  },
  captureButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonDisabled: {
    opacity: 0.4,
  },
  captureInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#fff',
  },
})
