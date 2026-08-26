.PHONY: help build build-release test clippy clippy-fix fmt fmt-check clean check-all ohos-build

# Default target
help:
	@echo "hiai-rs Makefile"
	@echo "================"
	@echo ""
	@echo "Common targets:"
	@echo "  make build          - Build in debug mode (bindgen; links libhiai on OHOS)"
	@echo "  make build-release  - Build in release mode"
	@echo "  make test           - Run all tests"
	@echo "  make clippy         - Run clippy lints"
	@echo "  make fmt            - Format code"
	@echo "  make fmt-check      - Check code formatting"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make check-all      - Run all checks (fmt, clippy, test)"
	@echo ""
	@echo "OpenHarmony targets:"
	@echo "  make ohos-build     - Cross-compile for aarch64-unknown-linux-ohos (requires CANN_DDK + OHOS_SDK_NATIVE)"

# Build targets (default: CANN bindings)
build:
	cargo build

build-release:
	cargo build --release

# Test targets
test:
	cargo test

# Linting targets
clippy:
	cargo clippy --all-targets -- -D warnings

clippy-fix:
	cargo clippy --fix --all-targets --allow-dirty

# Formatting targets
fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

# Clean target
clean:
	cargo clean

# Run all checks (useful before committing)
check-all: fmt-check clippy test
	@echo "[OK] All checks passed!"

# OpenHarmony cross-compile (full build + link against libhiai)
ohos-build:
	@if [ -z "$(CANN_DDK)" ]; then \
	    echo "Error: CANN_DDK not set. export CANN_DDK=/path/to/CANN-Kit-next/ddk/"; \
	    exit 1; \
	fi
	cargo build --target aarch64-unknown-linux-ohos --release
