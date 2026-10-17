// Places the voiceover clips inside each scene from lines.json + durations.json.
// A line starts at its `at` (seconds) or right after the previous line plus GAP, whichever is later.
import durations from './durations.json';
import spec from './lines.json';

const FPS = 30;
const GAP = 0.3;

export type Placed = {id: string; text: string; start: number; frames: number; end: number};

type Line = {id: string; at: number; text: string};
const SCENES = (spec as unknown as {scenes: Record<string, Line[]>}).scenes;
const DUR = durations as Record<string, {seconds: number}>;

const placed: Record<string, Placed[]> = {};
const byId: Record<string, Placed> = {};
for (const [scene, lines] of Object.entries(SCENES)) {
  let t = 0;
  placed[scene] = lines.map((l) => {
    const secs = DUR[l.id]?.seconds ?? 0;
    const start = Math.max(l.at, t === 0 ? 0 : t + GAP);
    t = start + secs;
    const p = {id: l.id, text: l.text, start: Math.round(start * FPS), frames: Math.ceil(secs * FPS), end: Math.ceil(t * FPS)};
    byId[l.id] = p;
    return p;
  });
}

export const sceneVoice = (scene: string): Placed[] => placed[scene] ?? [];
export const voiceEnd = (scene: string) => Math.max(0, ...sceneVoice(scene).map((l) => l.end));
export const lineStart = (id: string) => byId[id]?.start ?? 0;
export const lineEnd = (id: string) => byId[id]?.end ?? 0;
