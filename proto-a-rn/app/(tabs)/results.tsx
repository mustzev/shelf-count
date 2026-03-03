import { useLocalSearchParams, useRouter } from 'expo-router'
import { useEffect, useState } from 'react'
import {
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  useWindowDimensions,
  View,
} from 'react-native'
import { BoundingBoxOverlay } from '../../components/BoundingBoxOverlay'
import { clearResult, getResult } from '../../lib/frameStore'

export default function ResultsScreen() {
  const { photoPath } = useLocalSearchParams<{ photoPath: string }>()
  const router = useRouter()
  const { width: viewWidth } = useWindowDimensions()
  const [imageDims, setImageDims] = useState<{
    width: number
    height: number
  } | null>(null)

  const result = getResult()
  const photoUri = photoPath ? `file://${photoPath}` : null
  const viewHeight = imageDims
    ? viewWidth * (imageDims.height / imageDims.width)
    : viewWidth

  useEffect(() => {
    return () => {
      clearResult()
    }
  }, [])

  return (
    <View style={styles.container}>
      {photoUri && (
        <View style={{ width: viewWidth, height: viewHeight }}>
          <Image
            source={{ uri: photoUri }}
            style={{ width: viewWidth, height: viewHeight }}
            resizeMode="contain"
            onLoad={(e) => {
              const { width, height } = e.nativeEvent.source
              setImageDims({ width, height })
            }}
          />
          {result && imageDims && (
            <BoundingBoxOverlay
              detections={result.detections}
              imageWidth={imageDims.width}
              imageHeight={imageDims.height}
              viewWidth={viewWidth}
              viewHeight={viewHeight}
            />
          )}
        </View>
      )}

      <View style={styles.infoPanel}>
        {!result && (
          <Text style={styles.error}>
            No detection results. Please go back and try again.
          </Text>
        )}
        {result && (
          <>
            <Text style={styles.count}>{result.count} items detected</Text>
            <Text style={styles.timing}>
              Inference: {Math.round(result.inferenceTimeMs)}ms
            </Text>
          </>
        )}
      </View>

      <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
        <Text style={styles.backText}>Scan Another Shelf</Text>
      </TouchableOpacity>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  infoPanel: {
    padding: 20,
    alignItems: 'center',
  },
  count: {
    color: '#fff',
    fontSize: 32,
    fontWeight: '700',
    marginBottom: 8,
  },
  timing: {
    color: '#888',
    fontSize: 14,
  },
  error: {
    color: '#FF4444',
    fontSize: 14,
    textAlign: 'center',
  },
  backButton: {
    position: 'absolute',
    bottom: 40,
    alignSelf: 'center',
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  backText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
})
