# results/

Per-experiment outputs. Layout:

```
results/
├── S2-ahcptq/<model>/<method>/<seed>/{state.pt,eval.json,logs/}
├── S4-dbaf-weak/<model>/<baseline>/<seed>/{state.pt,eval.json}
├── S5-kv-pcsa/<model>/<context_len>/<seed>/{state.pt,eval.json}
├── S6-int4/<model>/<backend>/{packed_weights/,bench.json}
├── S7-ablations/<exp>/<model>/<seed>/{state.pt,eval.json}
├── S8-compsrt/<model>/<method>/<seed>/{state.pt,eval.json}
├── S9-downstream/<task>/<model>/<seed>/eval.json
└── deploy/<model>/<format>/{config.json,model.safetensors}
```

eval.json schema: `{model, bits, method, task, metric, value, seed, timestamp}`.
