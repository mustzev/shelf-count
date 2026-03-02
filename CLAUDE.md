# Shelf Count

Retail shelf audit system — photograph store shelves, count products, detect out-of-stocks.

## Project Structure

Monorepo with three prototypes exploring different architectures:

```
shelf-count/
├── docs/specs/          # Specifications (source of truth)
├── proto-a-rn/          # Prototype A: React Native + on-device ML (TFLite)
├── proto-b-flutter/     # Prototype B: Flutter + on-device ML (Google ML Kit)
├── proto-c-rust-flutter/ # Prototype C: Rust Axum server (ML) + Flutter client
```

## Development Approach

**Spec-driven development.** Every feature starts as a spec in `docs/specs/` before any code is written. Specs are the source of truth.

### Workflow
1. Write or update spec in `docs/specs/`
2. Implement against the spec
3. Verify implementation matches spec
4. Update spec if requirements change (spec stays in sync with reality)

## Conventions

- **Language:** TypeScript (proto-a), Dart (proto-b, proto-c client), Rust (proto-c server)
- **Commits:** Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`)
- **Prototype prefix:** Prefix commit messages with prototype scope when relevant, e.g. `feat(proto-a): add camera capture`
- **Specs:** Markdown files in `docs/specs/`. Use clear sections: Overview, Requirements, API/Interface, Data Flow, Acceptance Criteria
- **No premature abstraction:** Each prototype is independent. Do not share code between prototypes.
