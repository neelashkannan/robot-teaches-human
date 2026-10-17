// Render chosen frames of a composition to PNGs with one bundle: node scripts/stills.mjs Pitch 0 120 300 ...
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition} from '@remotion/renderer';
import path from 'node:path';

const [id = 'Pitch', ...frames] = process.argv.slice(2);
const serveUrl = await bundle({entryPoint: path.resolve('src/index.ts')});
const composition = await selectComposition({serveUrl, id});
for (const f of frames.length ? frames : ['0']) {
  const output = `out/stills/${id}-${String(f).padStart(4, '0')}.png`;
  await renderStill({serveUrl, composition, frame: Number(f), output, scale: 0.5});
  console.log(output);
}
