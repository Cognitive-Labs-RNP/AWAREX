#include <Pakhi1-project-1_inferencing.h>

/*
  AWAREX v5 - Driver Drowsiness + Distraction + Phone Monitor (ESP32, USB serial)

  Board    : ESP32 Dev Module (Arduino IDE)
  Libraries: Adafruit MPU6050, Adafruit Unified Sensor, Adafruit BusIO
  Wiring   : MPU6050  VCC->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22
             Buzzer   GPIO25 (through your transistor)
             Calibrate button GPIO4 <-> GND, Reset button GPIO18 <-> GND

  Fusion   : NORMAL    = 70% webcam + 30% MPU6050   (alarm at fused score >= 0.65)
             FALLBACK  = 40% webcam + 60% MPU6050   (glasses / low light, alarm >= 0.55)
             CAM_ONLY  = webcam only (no MPU or not calibrated)   (alarm >= 0.70)
             MPU_ONLY  = MPU only (no face)                       (alarm >= 0.70)

  Laptop -> ESP32 : D,<ear>,<face 0/1>,<yaw>,<pitch>,<lowVis 0/1>,<phone 0/1/2>\n
                      phone: 0 = none, 1 = phone at ear (talking), 2 = phone in front / low (using)
                      yaw: + = driver looks RIGHT, pitch: + = looking DOWN (laptop already
                      subtracts the neutral pose measured during calibration)
                    C\n = start calibration      R\n = reset alarms
  ESP32 -> Laptop : S,<situation>,<level 0/1/2>,<drowsyScore>\n   every 500 ms
                    P,1 / P,2   calibration phase started (1 = sit straight, 2 = nod forward)
                    K,1 / K,0 / K,2   calibration result (1 = done, 0 = failed, 2 = camera-only done)
                    F,...  feature line for TinyML data logging (every 200 ms)
                    M,...  debug line (every 500 ms)

  Priority: DROWSY > PHONE > DISTRACTED > PARTIALLY DROWSY
*/

#include <Wire.h>
#include <Adafruit_MPU6050.h>

// ====================== OPTIONS ======================
#define USE_MPU       1    // 1 = camera + MPU6050, 0 = camera only
#define USE_GYRO      1    // 1 = fuse gyro with accelerometer (stable in a moving car)
#define DEBUG_LINE    1    // 1 = send "M,..." debug line every 500 ms
#define FEATURE_LINE  1    // 1 = send "F,..." feature line every 200 ms (for TinyML data)

// ====================== PINS ======================
#define SDA_PIN         21
#define SCL_PIN         22
#define BUZZER_PIN      25
#define BTN_CAL_PIN      4
#define BTN_RESET_PIN   18
#define BUZZER_ON     HIGH
#define BUZZER_OFF    LOW
#define MPU_ADDR      0x68

// ====================== FUSION WEIGHTS AND TRIGGERS ======================
const float W_CAM_NORMAL   = 0.70;
const float W_MPU_NORMAL   = 0.30;
const float THR_NORMAL     = 0.65;

const float W_CAM_FALLBACK = 0.40;   // glasses / low light: EAR is less reliable
const float W_MPU_FALLBACK = 0.60;
const float THR_FALLBACK   = 0.55;

const float THR_CAM_ONLY   = 0.70;
const float THR_MPU_ONLY   = 0.70;

const float PARTIAL_BELOW  = 0.20;   // partial alarm = drowsy trigger - 0.20
const float HYST           = 0.10;   // alarm stays on until score drops this far below trigger
const float FLOAT_EPS      = 0.005;
const float HALF_SCORE_CAP = 0.60;   // half-closed eyes alone can never reach a drowsy alarm
const float NOD_NO_EYE_GATE = 0.40;  // head-down WITHOUT recent low eyes counts only 40% as a nod

// ====================== EYE THRESHOLDS (ratio = smoothed EAR / open-eye baseline) ======================
const float CLOSED_RATIO = 0.50;
const float HALF_RATIO   = 0.70;
const float EAR_MIN_OPEN = 0.20;
const float EAR_MAX_OPEN = 0.40;
const float EAR_START    = 0.27;

// ====================== TIMING ======================
const unsigned long CLOSE_FULL_MS     = 2000;  // eyes CLOSED this long -> camera drowsy score 1.0
const unsigned long HALF_FULL_MS      = 3000;  // eyes HALF/CLOSED this long -> half score reaches its cap
const unsigned long PARTIAL_ALARM_MS  = 1000;  // eyes low this long -> partial alarm
const unsigned long AWAY_FULL_MS      = 2000;  // looking away this long -> camera distraction score 1.0
const unsigned long STRONG_AFTER_MS   = 3000;  // alarm active this long -> stronger alarm
const unsigned long MPU_HOLD_MS       = 1000;  // head angle held this long -> full MPU score
const unsigned long CAMERA_TIMEOUT_MS = 3000;  // no camera data this long -> camera lost
const unsigned long NO_FACE_MS        = 2500;  // no face this long -> NO FACE
const unsigned long LOST_AWAY_MS      = 8000;  // face lost while head was turned away -> keep DISTRACTED this long
const unsigned long EYE_LINK_MS       = 3000;  // head-down is a "nod" only if eyes were low in the last 3 s
const unsigned long BASE_FAST_MS      = 8000;  // after boot / calibration the EAR baseline adapts quickly

const unsigned long PHONE_TALK_MS     = 1500;  // phone at ear this long -> alarm
const unsigned long PHONE_USE_MS      = 1200;  // phone in front + head down (or face lost) -> alarm
const unsigned long PHONE_USE_SLOW_MS = 2500;  // phone in front, head up -> alarm
const unsigned long PHONE_GRACE_MS    = 600;   // detector dropouts shorter than this are ignored
const float PHONE_PITCH_MIN           = 12.0;

// ====================== HEAD POSE LIMITS (from webcam, neutral already removed) ======================
const float YAW_LIMIT        = 30.0;
const float PITCH_DOWN_LIMIT = 20.0;
const float PITCH_UP_LIMIT   = 15.0;
const float EYE_YAW_MAX      = 22.0;   // head turned more than this: EAR is not reliable
const float EYE_PITCH_MAX    = 15.0;   // (looking down 15+ degrees makes eyelids look closed)
const float EYE_WIDE_YAW      = 50.0;   // beyond EYE_YAW_MAX: still check for CLOSED eyes up to this yaw
const float EYE_WIDE_DOWN     = 45.0;   // ...and up to this pitch down
const float EYE_WIDE_UP       = 35.0;   // ...and up to this pitch up
const float WIDE_CLOSED_RATIO = 0.35;   // off-axis eyes count as closed only below this (stricter than 0.50)
const unsigned long CLOSED_SUPPRESS_MS = 500;  // eyes closed this long -> head pose is not counted as distraction
const float LEARN_YAW_MAX    = 18.0;   // EAR baseline learns only when the head is frontal
const float LEARN_PITCH_MAX  = 12.0;

// ====================== MPU6050 ANGLE RANGES (neck mounted) ======================
const float NOD_START = 8.0;
const float NOD_FULL  = 25.0;
const float TILT_START = 10.0;
const float TILT_FULL  = 25.0;
const float MPU_DIR_SIDE = 15.0;
const float MPU_DIR_UP   = 12.0;
const float MIN_NOD_CAL_DEG = 10.0;
const float SIDE_SIGN = 1.0;           // set to -1.0 if LEFT and RIGHT are reversed
const float GYRO_ALPHA = 0.97;         // trust in gyro (higher = smoother, slower to correct drift)

// ====================== NAMES AND CODES ======================
#define ST_ATTENTIVE   0
#define ST_PARTIAL     1
#define ST_DROWSY      2
#define ST_DISTRACTED  3
#define ST_NO_FACE     4
#define ST_NO_CAMERA   5
#define ST_NO_DATA     6
#define ST_CALIBRATING 7
#define ST_PHONE_TALK  8
#define ST_PHONE_USE   9
const char* stateNames[] = {"ATTENTIVE", "PARTIALLY DROWSY", "DROWSY", "DISTRACTED",
                            "NO FACE", "NO CAMERA", "NO DATA", "CALIBRATING",
                            "PHONE TALKING", "PHONE USING"};

#define DIR_FRONT 0
#define DIR_LEFT  1
#define DIR_RIGHT 2
#define DIR_DOWN  3
#define DIR_UP    4
const char* dirNames[] = {"FRONT", "LEFT", "RIGHT", "DOWN", "UP"};

#define MODE_NORMAL   0
#define MODE_FALLBACK 1
#define MODE_CAM_ONLY 2
#define MODE_MPU_ONLY 3
#define MODE_NONE     4
const char* modeNames[] = {"NORMAL", "GLASSES_LOWLIGHT", "CAM_ONLY", "MPU_ONLY", "NONE"};

#define EYE_OPEN    0
#define EYE_HALF    1
#define EYE_CLOSED  2
#define EYE_UNKNOWN 3
const char* eyeNames[] = {"OPEN", "HALF", "CLOSED", "NA"};

// ====================== OBJECTS ======================
Adafruit_MPU6050 mpu;

// ====================== DATA FROM LAPTOP ======================
float camEar = 0;
float earS = 0.27;                   // smoothed EAR
float earBase = EAR_START;           // open-eye EAR tracked on the ESP32
float earRatio = 0;
bool  earInit = false;
unsigned long baseFastUntil = 0;
float yawDeg = 0;
float pitchDeg = 0;
bool  faceFound = false;
bool  lowVisFlag = false;
int   phoneZone = 0;                 // 0 none, 1 at ear, 2 in front / low
unsigned long lastCamMs = 0;
unsigned long lastFaceMs = 0;
bool  camOk = false;
bool  faceValidNow = false;

char rxBuf[80];
int  rxLen = 0;

// ====================== MPU6050 DATA ======================
bool  mpuOk = false;
bool  mpuCalibrated = false;
bool  mpuFirst = true;
float ax = 0, ay = 0, az = 0;        // filtered "up" vector in sensor frame
unsigned long lastImuUs = 0;
float U0[3]   = {0, 0, 1};
float FWD[3]  = {1, 0, 0};
float SIDE[3] = {0, 1, 0};
float U0n[3];
float sum0[3], sum1[3];
int   n0 = 0, n1 = 0;
int   calPhase = 0;                  // 0 = no, 1 = sit straight, 2 = nod forward
unsigned long calPhaseStart = 0;
float fwdDeg = 0;                    // + = leaning forward
float sideDeg = 0;                   // + = leaning right

// ====================== DECISION STATE ======================
unsigned long lowEyeSince = 0, closedSince = 0, awaySince = 0, nodSince = 0, tiltSince = 0;
unsigned long lastLowEyeMs = 0;
unsigned long drowsyOnSince = 0, distractOnSince = 0;
unsigned long talkSince = 0, talkSeenMs = 0, useSince = 0, useSeenMs = 0;
unsigned long useNeedMs = PHONE_USE_MS;
bool  drowsyOn = false, partialFusedOn = false, partialOn = false, distractOn = false;
bool  phoneTalkOn = false, phoneUseOn = false;
float scoreD = 0, scoreA = 0;
float activeThr = THR_NORMAL;
int   state = ST_NO_DATA;
int   alarmLevel = 0;
int   dirNow = DIR_FRONT;
int   lastAwayDir = DIR_FRONT;
int   mode = MODE_NONE;
int   eyeLevel = EYE_UNKNOWN;

// ====================== BUZZER CUES AND BUTTONS ======================
int cueCount = 0;
unsigned long cueStart = 0;

bool calStable = HIGH, calLastRaw = HIGH;
bool rstStable = HIGH, rstLastRaw = HIGH;
unsigned long calChangeMs = 0, rstChangeMs = 0;

// ====================== TIMERS ======================
unsigned long lastMpuMs = 0, lastMpuCheckMs = 0, lastDecisionMs = 0, lastUsbMs = 0, lastFeatMs = 0;

// ====================================================================
//                          SMALL HELPERS
// ====================================================================
unsigned long heldFor(unsigned long since) {
  return (since == 0) ? 0 : (millis() - since);
}

float clamp01(float x) {
  return (x < 0) ? 0 : ((x > 1) ? 1 : x);
}

// alarm switch with hysteresis: turns on at the trigger, turns off a bit lower
bool latch(bool wasOn, float score, float thr) {
  if (score <= 0.0f) return false;
  return wasOn ? (score >= thr - HYST) : (score >= thr - FLOAT_EPS);
}

float dot3(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void cross3(const float* a, const float* b, float* o) {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
}

bool normalize3(float* v) {
  float n = sqrt(dot3(v, v));
  if (n < 0.000001) return false;
  v[0] /= n; v[1] /= n; v[2] /= n;
  return true;
}

void cueBeeps(int n) {
  cueCount = n;
  cueStart = millis();
}

void resetDecisionState() {
  lowEyeSince = closedSince = awaySince = nodSince = tiltSince = 0;
  lastLowEyeMs = 0;
  drowsyOnSince = distractOnSince = 0;
  talkSince = talkSeenMs = useSince = useSeenMs = 0;
  drowsyOn = partialFusedOn = partialOn = distractOn = false;
  phoneTalkOn = phoneUseOn = false;
  scoreD = scoreA = 0;
  lastAwayDir = DIR_FRONT;
  alarmLevel = 0;
  state = ST_ATTENTIVE;
}

// ====================================================================
//                          CALIBRATION
// ====================================================================
void startCalibration() {
  calPhase = 1;
  calPhaseStart = millis();
  for (int i = 0; i < 3; i++) { sum0[i] = 0; sum1[i] = 0; }
  n0 = 0; n1 = 0;
  baseFastUntil = millis() + BASE_FAST_MS;   // relearn your open-eye EAR quickly
  resetDecisionState();
  cueBeeps(1);
  Serial.println("P,1");
}

void calFailed() {
  calPhase = 0;
  cueBeeps(5);
  resetDecisionState();
  Serial.println("K,0");
}

void finishNodCalibration() {
  calPhase = 0;
  if (!mpuOk || n1 < 5) { calFailed(); return; }

  float g1[3] = { sum1[0] / n1, sum1[1] / n1, sum1[2] / n1 };
  if (!normalize3(g1)) { calFailed(); return; }

  float c = constrain(dot3(U0n, g1), -1.0f, 1.0f);
  float angle = acos(c) * 57.2958f;
  if (angle < MIN_NOD_CAL_DEG) { calFailed(); return; }

  float d[3] = { g1[0] - c * U0n[0], g1[1] - c * U0n[1], g1[2] - c * U0n[2] };
  normalize3(d);

  for (int i = 0; i < 3; i++) { U0[i] = U0n[i]; FWD[i] = d[i]; }
  cross3(U0, FWD, SIDE);
  normalize3(SIDE);

  mpuCalibrated = true;
  cueBeeps(3);
  resetDecisionState();
  Serial.println("K,1");
}

void updateCalibration() {
  if (calPhase == 0) return;
  unsigned long el = millis() - calPhaseStart;

  if (calPhase == 1) {
    if (mpuOk && el >= 1000) {
      sum0[0] += ax; sum0[1] += ay; sum0[2] += az; n0++;
    }
    if (el >= 3000) {
      if (mpuOk && n0 > 5) {
        U0n[0] = sum0[0] / n0; U0n[1] = sum0[1] / n0; U0n[2] = sum0[2] / n0;
        normalize3(U0n);
        calPhase = 2;
        calPhaseStart = millis();
        cueBeeps(2);
        Serial.println("P,2");
      } else if (mpuOk) {
        calFailed();
      } else {
        calPhase = 0;                // no MPU: only the camera baseline is relearned
        cueBeeps(3);
        resetDecisionState();
        Serial.println("K,2");
      }
    }
  } else {
    if (el >= 2000) {
      sum1[0] += ax; sum1[1] += ay; sum1[2] += az; n1++;
    }
    if (el >= 4000) finishNodCalibration();
  }
}

// ====================================================================
//                          SERIAL FROM LAPTOP
// ====================================================================
void updateEarBaseline() {
  // learn only when the face looks at the camera and the eyes are not closing
  if (!faceFound) return;
  if (fabs(yawDeg) > LEARN_YAW_MAX || fabs(pitchDeg) > LEARN_PITCH_MAX) return;
  if (earS < EAR_MIN_OPEN * 0.6f) return;

  bool fast = millis() < baseFastUntil;
  if (earS > earBase) earBase += (fast ? 0.30f : 0.05f) * (earS - earBase);   // rises quickly
  else if (earS >= 0.6f * earBase)                                            // falls very slowly
    earBase -= (fast ? 0.05f : 0.002f) * (earBase - earS);
  earBase = constrain(earBase, EAR_MIN_OPEN, EAR_MAX_OPEN);
}

void parseLine(char* line) {
  if (line[0] == 'C') { startCalibration(); return; }
  if (line[0] == 'R') { resetDecisionState(); cueBeeps(1); return; }
  if (line[0] != 'D') return;

  strtok(line, ",");                       // skip the letter D
  char* t1 = strtok(NULL, ",");            // ear
  char* t2 = strtok(NULL, ",");            // face found
  char* t3 = strtok(NULL, ",");            // yaw
  char* t4 = strtok(NULL, ",");            // pitch
  char* t5 = strtok(NULL, ",");            // glasses / low light flag
  char* t6 = strtok(NULL, ",");            // phone zone
  if (!t1 || !t2 || !t3 || !t4 || !t5 || !t6) return;

  camEar     = atof(t1);
  faceFound  = atoi(t2) == 1;
  lowVisFlag = atoi(t5) == 1;
  phoneZone  = constrain(atoi(t6), 0, 2);
  lastCamMs  = millis();

  if (faceFound) {                                    // keep the last pose when the face is lost
    yawDeg   = atof(t3);
    pitchDeg = atof(t4);
  }

  if (faceFound && camEar > 0.01) {                   // ignore empty / zero readings
    if (!earInit) { earS = camEar; earInit = true; baseFastUntil = millis() + BASE_FAST_MS; }
    else earS = 0.6f * earS + 0.4f * camEar;
    updateEarBaseline();
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      rxBuf[rxLen] = 0;
      parseLine(rxBuf);
      rxLen = 0;
    } else if (c != '\r' && rxLen < (int)sizeof(rxBuf) - 1) {
      rxBuf[rxLen++] = c;
    }
  }
}

// ====================================================================
//                          MPU6050
// ====================================================================
void configureMpu() {
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  mpuFirst = true;
}

void checkMpu() {
  if (!USE_MPU) return;
  Wire.beginTransmission(MPU_ADDR);
  bool present = (Wire.endTransmission() == 0);

  if (present && !mpuOk) {
    if (mpu.begin(MPU_ADDR, &Wire)) {
      configureMpu();
      mpuOk = true;
      mpuCalibrated = false;
    }
  } else if (!present && mpuOk) {
    mpuOk = false;
    mpuCalibrated = false;
    fwdDeg = sideDeg = 0;
  }
}

void readMPU() {
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  unsigned long nowUs = micros();

  if (mpuFirst) {
    ax = a.acceleration.x; ay = a.acceleration.y; az = a.acceleration.z;
    lastImuUs = nowUs;
    mpuFirst = false;
    return;
  }

#if USE_GYRO
  // Complementary filter: turn the "up" vector with the gyro (fast, ignores car
  // braking / cornering), then pull it slowly towards the accelerometer (no drift).
  float dt = (nowUs - lastImuUs) * 1e-6f;
  lastImuUs = nowUs;
  if (dt > 0.2f) dt = 0.2f;
  float wx = g.gyro.x, wy = g.gyro.y, wz = g.gyro.z;       // rad/s
  float px = ax - (wy * az - wz * ay) * dt;                // u' = u - (w x u) * dt
  float py = ay - (wz * ax - wx * az) * dt;
  float pz = az - (wx * ay - wy * ax) * dt;
  ax = GYRO_ALPHA * px + (1.0f - GYRO_ALPHA) * a.acceleration.x;
  ay = GYRO_ALPHA * py + (1.0f - GYRO_ALPHA) * a.acceleration.y;
  az = GYRO_ALPHA * pz + (1.0f - GYRO_ALPHA) * a.acceleration.z;
#else
  lastImuUs = nowUs;
  ax = 0.8 * ax + 0.2 * a.acceleration.x;
  ay = 0.8 * ay + 0.2 * a.acceleration.y;
  az = 0.8 * az + 0.2 * a.acceleration.z;
#endif
}

void updateAngles() {
  if (!mpuOk || !mpuCalibrated) { fwdDeg = 0; sideDeg = 0; return; }
  float a[3] = { ax, ay, az };
  float dU = dot3(a, U0);
  float dF = dot3(a, FWD);
  float dS = dot3(a, SIDE);
  fwdDeg  = atan2(dF, dU) * 57.2958f;
  sideDeg = -atan2(dS, dU) * 57.2958f * SIDE_SIGN;
}

// ====================================================================
//                          DECISION LOGIC
// ====================================================================
void updateDecision() {
  unsigned long now = millis();

  if (calPhase != 0) {
    state = ST_CALIBRATING;
    alarmLevel = 0;
    return;
  }

  // ---------- 1) which sources are available? ----------
  camOk = (lastCamMs != 0) && (now - lastCamMs < CAMERA_TIMEOUT_MS);
  if (camOk && faceFound) lastFaceMs = now;
  faceValidNow = camOk && (lastFaceMs != 0) && (now - lastFaceMs < NO_FACE_MS);
  // face vanished while the head was turned away (e.g. looking at the side window)
  bool lostAway = camOk && !faceFound && (lastAwayDir != DIR_FRONT)
                  && (lastFaceMs != 0) && (now - lastFaceMs < LOST_AWAY_MS);
  bool camUsable = faceValidNow || lostAway;
  bool mpuUsable = mpuOk && mpuCalibrated;

  if (camUsable && mpuUsable) mode = lowVisFlag ? MODE_FALLBACK : MODE_NORMAL;
  else if (camUsable)         mode = MODE_CAM_ONLY;
  else if (mpuUsable)         mode = MODE_MPU_ONLY;
  else                        mode = MODE_NONE;

  // ---------- 2) weights for this mode ----------
  float wCam = 0, wMpu = 0, thr = THR_NORMAL;
  switch (mode) {
    case MODE_NORMAL:   wCam = W_CAM_NORMAL;   wMpu = W_MPU_NORMAL;   thr = THR_NORMAL;   break;
    case MODE_FALLBACK: wCam = W_CAM_FALLBACK; wMpu = W_MPU_FALLBACK; thr = THR_FALLBACK; break;
    case MODE_CAM_ONLY: wCam = 1.0;            wMpu = 0;              thr = THR_CAM_ONLY; break;
    case MODE_MPU_ONLY: wCam = 0;              wMpu = 1.0;            thr = THR_MPU_ONLY; break;
    default: break;
  }
  activeThr = thr;

  // ---------- 3) eye state (only when the head is frontal enough) ----------
  bool eyesVisible = camOk && faceFound;
   bool eyeReliable = eyesVisible && fabs(yawDeg) < EYE_YAW_MAX && fabs(pitchDeg) < EYE_PITCH_MAX;
  bool eyeWide = eyesVisible && !eyeReliable && fabs(yawDeg) < EYE_WIDE_YAW
                 && pitchDeg < EYE_WIDE_DOWN && pitchDeg > -EYE_WIDE_UP;
  earRatio = 0;
  if (eyeReliable) {
    earRatio = earS / earBase;
    if (earRatio < CLOSED_RATIO)    eyeLevel = EYE_CLOSED;
    else if (earRatio < HALF_RATIO) eyeLevel = EYE_HALF;
    else                            eyeLevel = EYE_OPEN;
  } else if (eyeWide) {                                  // head down / up / side: only a clearly closed eye counts
    earRatio = earS / earBase;
    eyeLevel = (earRatio < WIDE_CLOSED_RATIO) ? EYE_CLOSED : EYE_UNKNOWN;
  } else {
    eyeLevel = EYE_UNKNOWN;
  }
  bool eyesLow = (eyeLevel == EYE_HALF || eyeLevel == EYE_CLOSED);
  if (eyesLow) { if (lowEyeSince == 0) lowEyeSince = now; lastLowEyeMs = now; }
  else         { lowEyeSince = 0; }
  if (eyeLevel == EYE_CLOSED) { if (closedSince == 0) closedSince = now; }
  else                        { closedSince = 0; }
  unsigned long lowMs    = heldFor(lowEyeSince);
  unsigned long closedMs = heldFor(closedSince);

  // closed eyes ramp to 1.0, half-closed eyes ramp only to HALF_SCORE_CAP
  float camClosed = clamp01((float)closedMs / CLOSE_FULL_MS);
  float camHalf   = HALF_SCORE_CAP * clamp01((float)lowMs / HALF_FULL_MS);
  float camDrowsy = (camClosed > camHalf) ? camClosed : camHalf;

  // ---------- 4) MPU6050 scores ----------
  bool eyeLinked = (lastLowEyeMs != 0) && (now - lastLowEyeMs < EYE_LINK_MS);

   // with low eyes, head down / up / sideways all count as "head resting"; without low eyes only forward counts
  float restAmt = (fabs(fwdDeg) > fabs(sideDeg)) ? fabs(fwdDeg) : fabs(sideDeg);
  float nodAmt  = eyeLinked ? restAmt : fwdDeg;
  bool nodding = mpuUsable && (nodAmt >= NOD_START);
  if (nodding) { if (nodSince == 0) nodSince = now; }
  else         { nodSince = 0; }
  float mpuDrowsy = 0;
  if (mpuUsable) {
    mpuDrowsy = clamp01((nodAmt - NOD_START) / (NOD_FULL - NOD_START))
              * clamp01((float)heldFor(nodSince) / MPU_HOLD_MS);
    // head down while the eyes were open = looking down (phone / dashboard), not a nod
    if (mode == MODE_NORMAL && !eyeLinked) mpuDrowsy *= NOD_NO_EYE_GATE;
  }

  float tiltAmt = 0;
  if (mpuUsable) {
    float sideAmt = fabs(sideDeg);
    // forward lean counts as distraction only in NORMAL mode when it is NOT a nod
    float fwdAmt = (mode == MODE_NORMAL && !eyeLinked) ? fabs(fwdDeg) : -fwdDeg;
    tiltAmt = (fwdAmt > sideAmt) ? fwdAmt : sideAmt;
  }
  bool tilted = mpuUsable && (tiltAmt >= TILT_START);
  if (tilted) { if (tiltSince == 0) tiltSince = now; }
  else        { tiltSince = 0; }
  float mpuDist = 0;
  if (mpuUsable) {
    mpuDist = clamp01((tiltAmt - TILT_START) / (TILT_FULL - TILT_START))
            * clamp01((float)heldFor(tiltSince) / MPU_HOLD_MS);
  }

  // ---------- 5) where is the driver looking? (pose first, independent of the eyes) ----------
  int dir = DIR_FRONT;
  bool away = false;
  if (eyesVisible) {
    float exYaw  = fabs(yawDeg) / YAW_LIMIT;
    float exDown = (pitchDeg > 0) ? pitchDeg / PITCH_DOWN_LIMIT : 0;
    float exUp   = (pitchDeg < 0) ? -pitchDeg / PITCH_UP_LIMIT : 0;
    float best = 1.0;
    if (exYaw > best)  { best = exYaw;  dir = (yawDeg > 0) ? DIR_RIGHT : DIR_LEFT; }
    if (exDown > best) { best = exDown; dir = DIR_DOWN; }
    if (exUp > best)   { best = exUp;   dir = DIR_UP; }
    away = (dir != DIR_FRONT);
    lastAwayDir = dir;
  } else if (lostAway) {
    dir = lastAwayDir;
    away = true;
  }
  if (mode == MODE_MPU_ONLY) {
    if (sideDeg > MPU_DIR_SIDE)       dir = DIR_RIGHT;
    else if (sideDeg < -MPU_DIR_SIDE) dir = DIR_LEFT;
    else if (fwdDeg < -MPU_DIR_UP)    dir = DIR_UP;
  }
  dirNow = dir;

  if (away) { if (awaySince == 0) awaySince = now; }
  else      { awaySince = 0; }
  float camDist = clamp01((float)heldFor(awaySince) / AWAY_FULL_MS);

  // ---------- 6) fused scores ----------
    // eyes closed for a moment: the head pose is drowsiness, not distraction
  if (eyeLevel == EYE_CLOSED && closedMs >= CLOSED_SUPPRESS_MS) { camDist = 0; mpuDist = 0; }
  scoreD = wCam * camDrowsy + wMpu * mpuDrowsy;
  scoreA = wCam * camDist   + wMpu * mpuDist;
  float thrP = thr - PARTIAL_BELOW;

  // ---------- 7) alarms ----------
  bool d = (mode != MODE_NONE) && latch(drowsyOn, scoreD, thr);
  if (d && !drowsyOn) drowsyOnSince = now;
  if (!d) drowsyOnSince = 0;
  drowsyOn = d;

  partialFusedOn = (mode != MODE_NONE) && latch(partialFusedOn, scoreD, thrP);
  bool partialByTime = (mode == MODE_NORMAL || mode == MODE_CAM_ONLY)
                       && eyesLow && lowMs >= PARTIAL_ALARM_MS;     // not used with glasses / low light
  partialOn = partialFusedOn || partialByTime;

  bool da = (mode != MODE_NONE) && latch(distractOn, scoreA, thr);
  if (da && !distractOn) distractOnSince = now;
  if (!da) distractOnSince = 0;
  distractOn = da;

  // ---------- 7b) phone use (camera evidence only, tolerant to detector dropouts) ----------
  if (camOk && phoneZone == 1) { if (talkSince == 0) talkSince = now; talkSeenMs = now; }
  else if (talkSince != 0 && (now - talkSeenMs) > PHONE_GRACE_MS) talkSince = 0;

  if (camOk && phoneZone == 2) { if (useSince == 0) useSince = now; useSeenMs = now; }
  else if (useSince != 0 && (now - useSeenMs) > PHONE_GRACE_MS) useSince = 0;

  useNeedMs = (faceFound && pitchDeg < PHONE_PITCH_MIN) ? PHONE_USE_SLOW_MS : PHONE_USE_MS;
  phoneTalkOn = (talkSince != 0) && (heldFor(talkSince) >= PHONE_TALK_MS);
  phoneUseOn  = (useSince  != 0) && (heldFor(useSince)  >= useNeedMs);

  // ---------- 8) final state (drowsy > phone > distracted > partial) ----------
  alarmLevel = 0;
  if (drowsyOn) {
    state = ST_DROWSY;
    alarmLevel = (now - drowsyOnSince >= STRONG_AFTER_MS) ? 2 : 1;
  } else if (phoneTalkOn) {
    state = ST_PHONE_TALK;
    alarmLevel = (heldFor(talkSince) >= PHONE_TALK_MS + STRONG_AFTER_MS) ? 2 : 1;
  } else if (phoneUseOn) {
    state = ST_PHONE_USE;
    alarmLevel = (heldFor(useSince) >= useNeedMs + STRONG_AFTER_MS) ? 2 : 1;
  } else if (distractOn) {
    state = ST_DISTRACTED;
    alarmLevel = (now - distractOnSince >= STRONG_AFTER_MS) ? 2 : 1;
  } else if (partialOn) {
    state = ST_PARTIAL;
    alarmLevel = 1;
  } else if (mode == MODE_NONE) {
    state = ST_NO_DATA;
  } else if (!camUsable) {
    state = camOk ? ST_NO_FACE : ST_NO_CAMERA;
  } else {
    state = ST_ATTENTIVE;
  }
}

// ====================================================================
//                          BUZZER
// ====================================================================
void updateBuzzer() {
  unsigned long t = millis();
  bool on = false;

  if (cueCount > 0) {
    unsigned long el = t - cueStart;
    if (el >= (unsigned long)cueCount * 200) cueCount = 0;
    else on = (el % 200) < 100;
  } else if (calPhase != 0) {
    on = false;
  } else if (state == ST_DROWSY) {
    if (alarmLevel == 2) on = true;                    // strong: continuous tone
    else on = (t % 400) < 200;                         // fast beeps
  } else if (state == ST_PHONE_TALK) {
    unsigned long p = t % 1200;                        // three short beeps
    on = (p < 80) || (p >= 160 && p < 240) || (p >= 320 && p < 400);
    if (alarmLevel == 2) on = (t % 300) < 150;
  } else if (state == ST_PHONE_USE) {
    unsigned long p = t % 1200;                        // two long beeps
    on = (p < 250) || (p >= 400 && p < 650);
    if (alarmLevel == 2) on = (t % 500) < 300;
  } else if (state == ST_DISTRACTED) {
    if (alarmLevel == 2) {
      unsigned long p = t % 900;
      on = (p < 100) || (p >= 200 && p < 300) || (p >= 400 && p < 500);
    } else {
      unsigned long p = t % 1500;
      on = (p < 80) || (p >= 160 && p < 240);
    }
  } else if (state == ST_PARTIAL) {
    on = (t % 1500) < 100;                             // one short beep every 1.5 s
  }
  digitalWrite(BUZZER_PIN, on ? BUZZER_ON : BUZZER_OFF);
}

// ====================================================================
//                          BUTTONS
// ====================================================================
bool readButtonPress(int pin, bool &stableState, bool &lastRaw, unsigned long &changeMs) {
  bool raw = digitalRead(pin);
  unsigned long now = millis();
  if (raw != lastRaw) { lastRaw = raw; changeMs = now; }
  if ((now - changeMs) > 40 && raw != stableState) {
    stableState = raw;
    if (stableState == LOW) return true;
  }
  return false;
}

// ====================== TINYML (Edge Impulse) ======================
#define USE_TINYML 1
const float ML_CONF = 0.80;                                  // model must be at least this sure
#define ML_F 9                                               // features per sample (same as training)
#define ML_N (EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE / ML_F)     // samples per window (15)

static float mlWin[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
int mlFilled = 0;
const char* mlLabel = "-";
float mlConf = 0;
unsigned long lastMlPushMs = 0, lastMlRunMs = 0;

static int mlGetData(size_t offset, size_t length, float* out_ptr) {
  memcpy(out_ptr, mlWin + offset, length * sizeof(float));
  return 0;
}

// same 9 features, same order as the CSV: earRatio,yaw,pitch,fwd,side,face,mpuCal,lowVis,phone
void mlPushSample() {
  if (calPhase != 0) { mlFilled = 0; return; }
  memmove(mlWin, mlWin + ML_F, (ML_N - 1) * ML_F * sizeof(float));
  float* p = mlWin + (ML_N - 1) * ML_F;
  p[0] = earRatio;  p[1] = yawDeg;  p[2] = pitchDeg;  p[3] = fwdDeg;  p[4] = sideDeg;
  p[5] = faceFound ? 1 : 0;
  p[6] = (mpuOk && mpuCalibrated) ? 1 : 0;
  p[7] = lowVisFlag ? 1 : 0;
  p[8] = phoneZone;
  if (mlFilled < ML_N) mlFilled++;
}

void mlRun() {
  mlLabel = "-"; mlConf = 0;
  if (mlFilled < ML_N) return;                       // window not full yet (first 3 s)
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = &mlGetData;
  ei_impulse_result_t result;
  if (run_classifier(&signal, &result, false) != EI_IMPULSE_OK) return;
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    if (result.classification[i].value > mlConf) {
      mlConf = result.classification[i].value;
      mlLabel = result.classification[i].label;
    }
  }
}

// model can raise an alarm when it is confident; it never cancels a hardcode alarm
void applyMlFusion() {
  if (!USE_TINYML || mlConf < ML_CONF) return;
  if (state != ST_ATTENTIVE && state != ST_PARTIAL && state != ST_DISTRACTED) return;
  int s = -1;
  if      (!strcmp(mlLabel, "drowsy"))     s = ST_DROWSY;
  else if (!strcmp(mlLabel, "partial"))    s = ST_PARTIAL;
  else if (!strcmp(mlLabel, "distracted")) s = ST_DISTRACTED;
  else if (!strcmp(mlLabel, "phoneuse"))   s = ST_PHONE_USE;
  if (s < 0) return;                                 // "attentive": the hardcode result stays
  if (s == ST_PARTIAL && state == ST_DISTRACTED) return;   // never downgrade
  state = s;
  if (alarmLevel == 0) alarmLevel = 1;
}
// ====================================================================
//                          USB SERIAL OUTPUT TO LAPTOP
// ====================================================================
void buildSituation(char* out, size_t n) {
  if (state == ST_DISTRACTED) {
    snprintf(out, n, "DISTRACTED-LOOKING %s", dirNames[dirNow]);
  } else if ((state == ST_NO_FACE || state == ST_NO_CAMERA) && mode == MODE_MPU_ONLY) {
    snprintf(out, n, "%s - MPU ONLY", stateNames[state]);
  } else {
    snprintf(out, n, "%s", stateNames[state]);
  }
}

void sendUsbStatus() {
  char sit[48];
  buildSituation(sit, sizeof(sit));
  Serial.printf("S,%s,%d,%.2f\n", sit, alarmLevel, scoreD);
    Serial.printf("T,%s,%.2f\n", mlLabel, mlConf);
#if DEBUG_LINE
  Serial.printf("M,%s,eye=%s,ear=%.3f,base=%.3f,face=%d,mpuOk=%d,mpuCal=%d,phone=%d,thr=%.2f,score=%.2f\n",
                modeNames[mode], eyeNames[eyeLevel], earS, earBase,
                faceFound, mpuOk, mpuCalibrated, phoneZone, activeThr, scoreD);
#endif
}

void sendFeatureLine() {
  unsigned long ph = heldFor(talkSince);
  if (heldFor(useSince) > ph) ph = heldFor(useSince);
  Serial.printf("F,%.3f,%.1f,%.1f,%.1f,%.1f,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d\n",
                earRatio, yawDeg, pitchDeg, fwdDeg, sideDeg,
                faceFound ? 1 : 0, (mpuOk && mpuCalibrated) ? 1 : 0, lowVisFlag ? 1 : 0, phoneZone,
                heldFor(lowEyeSince) / 1000.0f, heldFor(closedSince) / 1000.0f,
                heldFor(awaySince) / 1000.0f, heldFor(nodSince) / 1000.0f,
                heldFor(tiltSince) / 1000.0f, ph / 1000.0f, mode, state);
}

// ====================================================================
//                          SETUP AND LOOP
// ====================================================================
void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, BUZZER_OFF);
  pinMode(BTN_CAL_PIN, INPUT_PULLUP);
  pinMode(BTN_RESET_PIN, INPUT_PULLUP);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  mpuOk = USE_MPU && mpu.begin(MPU_ADDR, &Wire);   // not fatal: no MPU = webcam-only
  if (mpuOk) configureMpu();

  baseFastUntil = millis() + BASE_FAST_MS;
  resetDecisionState();
  state = ST_NO_DATA;
  cueBeeps(1);
}

void loop() {
  readSerial();

  unsigned long now = millis();

  if (readButtonPress(BTN_CAL_PIN, calStable, calLastRaw, calChangeMs)) {
    startCalibration();                              // also sends "P,1" so the laptop shows CALIBRATING
  }
  if (readButtonPress(BTN_RESET_PIN, rstStable, rstLastRaw, rstChangeMs)) {
    resetDecisionState();
    cueBeeps(1);
  }

  if (now - lastMpuMs >= 20) {
    lastMpuMs = now;
    if (mpuOk) readMPU();
    updateCalibration();
    updateAngles();
  }
  if (now - lastMpuCheckMs >= 500) { lastMpuCheckMs = now; checkMpu(); }
    if (now - lastDecisionMs >= 100) { lastDecisionMs = now; updateDecision(); applyMlFusion(); }
#if USE_TINYML
  if (now - lastMlPushMs >= 200) { lastMlPushMs = now; mlPushSample(); }
  if (now - lastMlRunMs  >= 500) { lastMlRunMs  = now; mlRun(); }
#endif

  updateBuzzer();

  if (now - lastUsbMs >= 500) { lastUsbMs = now; sendUsbStatus(); }
#if FEATURE_LINE
  if (now - lastFeatMs >= 200) { lastFeatMs = now; sendFeatureLine(); }
#endif
}
