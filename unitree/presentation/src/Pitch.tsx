import React from 'react';
import {Audio, interpolate, staticFile} from 'remotion';
import {linearTiming, TransitionSeries} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';
import {slide} from '@remotion/transitions/slide';
import {Close, CLOSE_FRAMES} from './scenes/Close';
import {ARM_FRAMES, ArmDemo, UNITREE_FRAMES, UnitreeDemo} from './scenes/Demos';
import {Flip, FLIP_FRAMES} from './scenes/Flip';
import {Hype, HYPE_FRAMES} from './scenes/Hype';
import {Limits, LIMITS_FRAMES} from './scenes/Limits';
import {Next, NEXT_FRAMES} from './scenes/Next';
import {Stack, STACK_FRAMES} from './scenes/Stack';
import {Title, TITLE_FRAMES} from './scenes/Title';
import {Trick, TRICK_FRAMES} from './scenes/Trick';
import {Virtual, VIRTUAL_FRAMES} from './scenes/Virtual';
import {Why, WHY_FRAMES} from './scenes/Why';
import {VoiceTrack} from './components/VoiceTrack';
import {sceneVoice, voiceEnd} from './voice/voice';

const T = 15; // default crossfade length in frames

type Scene = {id: string; frames: number; C: React.FC; inT?: number; inSlide?: boolean};

// Story order follows the judging rubric: flip (creativity), demo (execution),
// how it works (understanding), stack and AI disclosure, honest limits, close.
// No robot access at the venue, so the virtual world is explained before the demos.
// The demos escalate: the small teacher (arm) first, a hype bridge, then the humanoid.
const BASE: Scene[] = [
  {id: 'Flip', frames: FLIP_FRAMES, C: Flip},
  {id: 'Title', frames: TITLE_FRAMES, C: Title},
  {id: 'Why', frames: WHY_FRAMES, C: Why},
  {id: 'Virtual', frames: VIRTUAL_FRAMES, C: Virtual},
  {id: 'ArmDemo', frames: ARM_FRAMES, C: ArmDemo},
  {id: 'Hype', frames: HYPE_FRAMES, C: Hype, inT: 12, inSlide: true},
  {id: 'UnitreeDemo', frames: UNITREE_FRAMES, C: UnitreeDemo, inT: 8},
  {id: 'Trick', frames: TRICK_FRAMES, C: Trick},
  {id: 'Stack', frames: STACK_FRAMES, C: Stack},
  {id: 'Limits', frames: LIMITS_FRAMES, C: Limits},
  {id: 'Next', frames: NEXT_FRAMES, C: Next},
  {id: 'Close', frames: CLOSE_FRAMES, C: Close},
];

// A scene lasts at least as long as its visuals, and long enough for its voice to finish
// (the voice may run into the crossfade, but never under the next scene's first line).
export const SCENES: Scene[] = BASE.map((s) => ({...s, frames: Math.max(s.frames, voiceEnd(s.id) + 10)}));

const STARTS = SCENES.reduce<number[]>((acc, x, i) => [...acc, i === 0 ? 0 : acc[i - 1] + SCENES[i - 1].frames - (x.inT ?? T)], []);
export const PITCH_FRAMES = STARTS[STARTS.length - 1] + SCENES[SCENES.length - 1].frames;

// Every voice line on the whole timeline, for ducking the music under it.
const SPEECH = SCENES.flatMap((s, i) => sceneVoice(s.id).map((l) => [STARTS[i] + l.start, STARTS[i] + l.end] as const));
export const HAS_MUSIC = true;
// Lyria 3 (google/lyria-3-pro-preview via OpenRouter), normalised to -18 LUFS. Ducks under the voice.
const MUSIC = {file: 'music/bed.mp3', level: 0.55, ducked: 0.16};
const musicVolume = (f: number) => {
  const talking = SPEECH.some(([a, b]) => f >= a - 8 && f <= b + 8);
  const fadeIn = interpolate(f, [0, 20], [0, 1], {extrapolateRight: 'clamp'});
  const fadeOut = interpolate(f, [PITCH_FRAMES - 60, PITCH_FRAMES], [1, 0], {extrapolateLeft: 'clamp'});
  return (talking ? MUSIC.ducked : MUSIC.level) * fadeIn * fadeOut;
};

export const Pitch: React.FC<{music?: boolean}> = ({music = HAS_MUSIC}) => (
  <>
  {music && <Audio src={staticFile(MUSIC.file)} volume={musicVolume} />}
  <TransitionSeries>
    {SCENES.flatMap(({id, frames, C, inT = T, inSlide}, i) => {
      const seq = (
        <TransitionSeries.Sequence key={id} durationInFrames={frames}>
          <C />
          <VoiceTrack scene={id} />
        </TransitionSeries.Sequence>
      );
      return i === 0
        ? [seq]
        : [
            <TransitionSeries.Transition
              key={`${id}-t`}
              presentation={inSlide ? slide({direction: 'from-right'}) : fade()}
              timing={linearTiming({durationInFrames: inT})}
            />,
            seq,
          ];
    })}
  </TransitionSeries>
  </>
);
