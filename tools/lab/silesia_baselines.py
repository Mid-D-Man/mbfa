import zlib, lzma, zstandard, json, os, time
import sys
base=sys.argv[1] if len(sys.argv)>1 else 'silesia'
names=["xml","ooffice","reymont","sao","x-ray","mr","osdb","dickens","samba","nci","webster","mozilla"]
out={}
zc=zstandard.ZstdCompressor(level=3)
for n in names:
    d=open(os.path.join(base,n),'rb').read()
    t=time.time(); g=len(zlib.compress(d,6)); tg=time.time()-t
    t=time.time(); z=len(zc.compress(d)); tz=time.time()-t
    t=time.time(); x=len(lzma.compress(d,preset=6)); tx=time.time()-t
    out[n]={"orig":len(d),"gzip":g,"zstd":z,"xz":x,"gzip_s":round(tg,1),"zstd_s":round(tz,1),"xz_s":round(tx,1)}
    json.dump(out,open('silesia_baselines.json','w'))
    print(n,out[n],flush=True)
print('done')
