import 'package:flutter_test/flutter_test.dart';

import 'package:shelf_count_client/main.dart';

void main() {
  testWidgets('ShelfCountApp smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const ShelfCountApp());
  });
}
