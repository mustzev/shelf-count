# iOS Camera Permission Setup

After running `flutter create`, add this to `ios/Runner/Info.plist` inside the top `<dict>`:

```xml
<key>NSCameraUsageDescription</key>
<string>Camera access is needed to photograph store shelves for product counting.</string>
```

This is required for the `camera` package to work on iOS.
