import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {Backdrop, Body, FadeUp, Headline, Label, ramp} from '../components/Anim';
import {C, FONT} from '../theme';

export const TITLE_FRAMES = 152;

// The rubric asks for the old assumption and the flip in one sentence each.
export const Title: React.FC = () => {
  const frame = useCurrentFrame();
  const strike = ramp(frame, 62, 80);
  return (
    <AbsoluteFill>
      <Backdrop glow={C.orange} />
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', flexDirection: 'column'}}>
        <FadeUp>
          <Headline size={150} style={{textAlign: 'center', letterSpacing: '-0.03em'}}>
            Robot <span style={{color: C.orange}}>Teaches</span> Human
          </Headline>
        </FadeUp>
        <div style={{marginTop: 70, display: 'grid', gridTemplateColumns: '300px 1100px', rowGap: 34, alignItems: 'baseline'}}>
          <FadeUp delay={30}>
            <Label color={C.muted} size={24}>
              Old assumption
            </Label>
          </FadeUp>
          <FadeUp delay={30}>
            <span style={{position: 'relative', fontFamily: FONT.display, fontWeight: 700, fontSize: 52, color: C.muted}}>
              Humans teach robots.
              <span
                style={{
                  position: 'absolute',
                  left: 0,
                  top: '55%',
                  height: 6,
                  width: `${strike * 100}%`,
                  background: C.red,
                  borderRadius: 3,
                }}
              />
            </span>
          </FadeUp>
          <FadeUp delay={84}>
            <Label size={24}>Our flip</Label>
          </FadeUp>
          <FadeUp delay={84}>
            <div style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 52, color: C.text, lineHeight: 1.12}}>
              The robot shows the move, watches you copy it, and corrects you limb by limb.
            </div>
          </FadeUp>
        </div>
        <FadeUp delay={128}>
          <Body size={26} style={{marginTop: 56, textAlign: 'center'}}>
            Edinburgh AI Hackathon · “Flip the Script”
          </Body>
        </FadeUp>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
