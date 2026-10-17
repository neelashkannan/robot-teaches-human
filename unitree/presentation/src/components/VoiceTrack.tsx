import React from 'react';
import {Audio, Sequence, staticFile} from 'remotion';
import {sceneVoice} from '../voice/voice';

export const VoiceTrack: React.FC<{scene: string}> = ({scene}) => (
  <>
    {sceneVoice(scene).map((l) => (
      <Sequence key={l.id} from={l.start} durationInFrames={l.frames + 3} layout="none">
        <Audio src={staticFile(`voice/${l.id}.wav`)} />
      </Sequence>
    ))}
  </>
);
