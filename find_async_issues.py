#!/usr/bin/env python3
"""Find async/sync method call mismatches in orchestrator."""

import re
from pathlib import Path

# Read orchestrator file
orch_path = Path("pipeline/orchestrator.py")
with open(orch_path) as f:
    orch_content = f.read()

# Find all await ctx.* calls
await_pattern = r'await ctx\.(\w+)\.(\w+)\('
matches = re.findall(await_pattern, orch_content)

print("Async method calls in orchestrator:\n")
for service, method in set(matches):
    print(f"  ctx.{service}.{method}()")

# Check if methods exist and are async
print("\nChecking actual method signatures:\n")

service_files = {
    'query_generator': 'services/query_generation.py',
    'search_service': 'services/search.py',
    'scraper': 'services/scraper.py',
    'event_extractor': 'services/event_extractor.py',
    'embedding_service': 'services/embedding.py',
    'clustering_service': 'services/clustering.py',
    'timeline_builder': 'services/timeline.py',
    'forecast_generator': 'services/forecast.py',
}

issues = []

for service, method in set(matches):
    if service in service_files:
        file_path = Path(service_files[service])
        if file_path.exists():
            with open(file_path) as f:
                content = f.read()

            # Check if method exists and is async
            async_pattern = rf'async def {method}\('
            sync_pattern = rf'def {method}\('

            has_async = bool(re.search(async_pattern, content))
            has_sync = bool(re.search(sync_pattern, content))

            if has_sync and not has_async:
                issues.append(f"  ❌ {service}.{method}() is SYNC but called with await")
                print(f"  ❌ {service}.{method}() is SYNC but called with await")
            elif has_async:
                print(f"  ✓ {service}.{method}() is async")
            else:
                issues.append(f"  ⚠️  {service}.{method}() NOT FOUND")
                print(f"  ⚠️  {service}.{method}() NOT FOUND")

if issues:
    print(f"\n{len(issues)} issues found that need fixing!")
else:
    print("\n✓ All async calls look good!")
