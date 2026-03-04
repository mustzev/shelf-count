import { StyleSheet, Text, View } from 'react-native'
import type { Detection } from '../lib/ml/types'

interface Props {
  detections: Detection[]
  imageWidth: number
  imageHeight: number
  viewWidth: number
  viewHeight: number
}

export function BoundingBoxOverlay({
  detections,
  imageWidth,
  imageHeight,
  viewWidth,
  viewHeight,
}: Props) {
  const scaleX = viewWidth / imageWidth
  const scaleY = viewHeight / imageHeight

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {detections.map((det) => (
        <View
          key={`${det.bbox.x}-${det.bbox.y}-${det.bbox.width}-${det.bbox.height}`}
          style={[
            styles.box,
            {
              left: det.bbox.x * imageWidth * scaleX,
              top: det.bbox.y * imageHeight * scaleY,
              width: det.bbox.width * imageWidth * scaleX,
              height: det.bbox.height * imageHeight * scaleY,
            },
          ]}
        >
          <Text style={styles.label} numberOfLines={1}>
            {Math.round(det.confidence * 100)}%
          </Text>
        </View>
      ))}
    </View>
  )
}

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00FF00',
    backgroundColor: 'transparent',
    overflow: 'visible',
  },
  label: {
    position: 'absolute',
    top: -16,
    left: 0,
    backgroundColor: '#00FF00',
    color: '#000',
    fontSize: 10,
    fontWeight: '700',
    paddingHorizontal: 2,
    paddingVertical: 1,
  },
})
