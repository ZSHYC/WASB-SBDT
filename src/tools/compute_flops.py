import os
import sys
from pathlib import Path

def dict_to_attr(d):
    # simple recursive AttrDict-like wrapper
    if isinstance(d, dict):
        class ADict(dict):
            def __getattr__(self, name):
                val = self.get(name)
                if isinstance(val, dict):
                    return dict_to_attr(val)
                return val
            def __setattr__(self, name, value):
                self[name] = value
        ad = ADict()
        for k, v in d.items():
            ad[k] = dict_to_attr(v) if isinstance(v, dict) else v
        return ad
    return d

def load_yaml_strip(path):
    import yaml
    s = open(path, 'r', encoding='utf-8').read()
    # remove ``` fences if present
    s = s.replace('```yaml', '').replace('```', '')
    return yaml.safe_load(s)

def main():
    # locate project src and add to path
    this = Path(__file__).resolve()
    src_dir = this.parents[1]
    sys.path.insert(0, str(src_dir))

    cfg_path = src_dir / 'configs' / 'model' / 'wasb.yaml'
    if not cfg_path.exists():
        print('cannot find', cfg_path)
        return

    cfg = load_yaml_strip(cfg_path)
    cfg = dict_to_attr(cfg)

    import torch
    from models.hrnet import HRNet

    model = HRNet(cfg)
    model.eval()

    # wrapper to return a single tensor (final scale 0)
    import torch.nn as nn
    class Wrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            out = self.net(x)
            # out is a dict mapping scales -> tensor
            return out[0]

    wrapped = Wrapper(model)

    C = 3 * int(cfg.frames_in)
    H = 1080
    W = 1920

    print('Input:', (C, H, W))

    # try ptflops
    try:
        from ptflops import get_model_complexity_info
    except Exception as e:
        print('ptflops not installed:', e)
        print('You can install with: pip install ptflops')
        get_model_complexity_info = None

    # avoid using torchsummary because HRNet uses list inputs in some modules
    if get_model_complexity_info is not None:
        # clear any existing forward hooks (torchsummary may have left hooks from previous runs)
        for m in wrapped.modules():
            if hasattr(m, '_forward_hooks'):
                try:
                    m._forward_hooks.clear()
                except Exception:
                    pass

        print('\n=== ptflops ===')
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            # print per-layer stat so we can export it
            macs, params = get_model_complexity_info(wrapped, (C, H, W), as_strings=True,
                                                     print_per_layer_stat=True, verbose=False)
        finally:
            sys.stdout = old_stdout

        printed = buf.getvalue()
        print(printed)

        # write per-layer table to CSV by parsing printed table
        lines = [l.rstrip() for l in printed.splitlines()]
        # find table start (line that starts with 'Name')
        table_start = None
        for i, l in enumerate(lines):
            if l.strip().startswith('Name') or l.strip().startswith('Layer'):
                table_start = i
                break

        csv_lines = []
        if table_start is not None:
            header = lines[table_start]
            # subsequent separator line usually consists of dashes
            # collect until an empty line or summary line
            j = table_start + 1
            # skip separator if present
            if j < len(lines) and set(lines[j].strip()) <= set('- '):
                j += 1
            csv_lines.append('layer,macs,params,extra')
            while j < len(lines):
                l = lines[j]
                if not l.strip():
                    break
                # try to split by multiple spaces
                parts = [p for p in l.split('  ') if p.strip()]
                if len(parts) >= 3:
                    layer = parts[0].strip()
                    mac = parts[1].strip()
                    par = parts[2].strip()
                    extra = ' '.join(parts[3:]) if len(parts) > 3 else ''
                    # escape commas in layer name
                    layer = layer.replace(',', ';')
                    csv_lines.append(f'{layer},{mac},{par},{extra}')
                j += 1

        out_dir = Path(__file__).resolve().parents[2] / 'reports'
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / 'wasb_flops_per_layer.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(csv_lines))

        # write summary markdown
        md_path = out_dir / 'wasb_flops_summary.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('# WASB (HRNet) FLOPs & Params Report\n\n')
            f.write(f'- Input: channels={C}, H={H}, W={W}\n')
            f.write(f'- Params: {params}\n')
            f.write(f'- MACs: {macs}\n')
            f.write('\n\n')
            f.write('Per-layer details saved in `wasb_flops_per_layer.csv`.\n')

        print('Wrote:', csv_path)
        print('Wrote:', md_path)
    else:
        # fallback: count params only
        total_params = sum(p.numel() for p in model.parameters())
        print('\nParams (count):', total_params)

if __name__ == '__main__':
    main()
