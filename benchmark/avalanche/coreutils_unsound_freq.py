import subprocess, re, collections, os
BINS = ['ls','cat','cp','mv','base64','sha256sum','md5sum','sort','wc','od','tr','cut','head','tail','du','dd','printf','comm','join','paste','fold','expand','nl','fmt','pr','shuf','factor','seq']
def mnem_stream(path):
    out = subprocess.run(['objdump','-d','--no-show-raw-insn',path], capture_output=True, text=True).stdout
    func=None
    for line in out.splitlines():
        m=re.match(r'^[0-9a-f]+ <([^>]+)>:', line)
        if m: func=m.group(1); continue
        m=re.match(r'^\s+[0-9a-f]+:\s+(\S+)', line)
        if m: yield func, m.group(1)
def cat(mn):
    if mn.startswith('cmov'): return 'cmov'
    if re.match(r'^set[a-z]', mn) and mn!='seta'*0: return 'setcc'  # all setcc start set<cc>
    if mn in ('adc','sbb','adcx','adox','adcl','adcq','sbbl','sbbq'): return 'carry'
    return None
tot=0; cnt=collections.Counter(); funcs=set(); hit_funcs=set()
percat=collections.Counter()
per_bin={}
for b in BINS:
    p=f'/usr/bin/{b}'
    if not os.path.exists(p): continue
    bt=0; bc=collections.Counter()
    for func,mn in mnem_stream(p):
        tot+=1; bt+=1; funcs.add((b,func))
        c=cat(mn)
        if c:
            cnt[c]+=1; bc[c]+=1; percat[mn]+=1; hit_funcs.add((b,func))
    per_bin[b]=(bt,sum(bc.values()))
target=cnt['cmov']+cnt['setcc']+cnt['carry']
print(f"binaries analysed: {len([b for b in BINS if os.path.exists(f'/usr/bin/{b}')])}")
print(f"total instructions: {tot:,}")
print(f"  cmov*   : {cnt['cmov']:,}  ({100*cnt['cmov']/tot:.2f}%)")
print(f"  setcc   : {cnt['setcc']:,}  ({100*cnt['setcc']/tot:.2f}%)")
print(f"  carry(adc/sbb/adcx/adox): {cnt['carry']:,}  ({100*cnt['carry']/tot:.2f}%)")
print(f"  COMBINED (unsound-triggering): {target:,}  ({100*target/tot:.2f}% of all instructions)")
print(f"functions total: {len(funcs):,}")
print(f"functions containing >=1 cmov/setcc/carry: {len(hit_funcs):,}  ({100*len(hit_funcs)/len(funcs):.1f}% of functions)")
print("top specific mnemonics:", dict(percat.most_common(12)))
