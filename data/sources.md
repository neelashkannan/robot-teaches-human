# Sign sources (hx-data, 17 Oct 2026)

How the handshapes were verified: BSL SignBank (https://bslsignbank.ucl.ac.uk/) lists the number signs
and regional variants but its sign pages are video only. They carry no text handshape, and its
handshape search needs a login or JavaScript, so the finger-by-finger descriptions come from the
dictionary at british-sign.co.uk (Lisia Digital Limited, a commercial BSL course site). It is NOT
SignBank, the BDA or Signature, and it is not a fluent-signer review. Each entry says "illustration is a
starting point, watch video". Spread and thumb values in `signs.json` are my estimates where the source
is silent (noted per sign).

## Used (in `signs.json`)
- one: https://www.british-sign.co.uk/bsl-dictionary/one ("Index finger extended"); SignBank https://bslsignbank.ucl.ac.uk/dictionary/regional/one-1.html
- two: https://www.british-sign.co.uk/bsl-dictionary/two ("Index and middle fingers extended"); SignBank .../regional/two-1.html
- three: https://www.british-sign.co.uk/bsl-dictionary/three ("Index, middle, and ring fingers extended"); SignBank .../regional/three-1.html
- five: https://www.british-sign.co.uk/bsl-dictionary/five ("All fingers and thumb extended"); SignBank .../regional/five-1.html
- seven: https://www.british-sign.co.uk/bsl-dictionary/seven ("Thumb and index finger extended - held horizontal"); SignBank .../regional/seven-1.html
- sorry: https://www.british-sign.co.uk/bsl-dictionary/sorry ("Closed hand makes a circular motion on chest"); not checked on SignBank

## Kept in `signs.json` as `not_representable`
- four ("Index, middle, ring, and little fingers extended"): needs the little finger; identical to three on this hand.
- nine ("Thumb, index, middle, and ring fingers extended - held horizontal"): identical to five here apart from orientation.

## Verified but not used (spare candidates, not in the JSON)
- six: "Closed hand with bent thumb held up." Thumb only differs from the fist in sorry; unreliable if the thumb is dropped.
- eight: "Thumb, index, and middle fingers extended - held horizontal." Thumb-dependent, orientation not checked.
- good: "Short forward movement using closed hand with thumb up." Same handshape as six, movement not checked.

## Rejected and why
- Fingerspelling letters (two-handed in BSL).
- father ("Fingerspelled F"), mother ("Fingerspelled M taps palm"), car, love: two hands.
- name ("N" shape touching forehead then twist), eat (bunched fingers), tea, bird, money, promise, snake, sleep: need the little finger, a pinch or flick, or both hands; location or movement is the sign.
- hello, nice: handshape is not the defining part (wave near head; thumb brushes chin).
- me, you: same handshape as one; the meaning is the pointing direction, which is not checked.
- ten: no handshape text found in the source.
