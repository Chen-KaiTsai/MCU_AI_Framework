#include "framework.h"

// Setup Timer
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
hw_timer_t* timer;

bool frameworkTaskFlag = true;
int count = 0;
framework::sequential model;
float* data = new float[1 * 3 * 16 * 16];
float* predict = new float[16 * 16 * 1];
std::vector<float*> weights;
float* conv2d_w = new float[(3 * 3 * 3 * 1) + 1];

void taskFramework(void* pvParam) {
  Serial.printf("Deploy Task : Framework\n");
  while (true) {
    if (frameworkTaskFlag == true) {
      Serial.printf("Start Task : Framework\n");
      model.inference(data, predict);

      for (int i = 0; i < (16 * 16 * 1); ++i)
        Serial.printf("[%d] : %+03.4f\n", i, predict[i]);
    }

    frameworkTaskFlag = false;
    vTaskDelay(2500);
  }
}

void IRAM_ATTR onTimerStatusReport() {
  // Print out free heap size
  portENTER_CRITICAL(&mux);
  Serial.printf("%u total heap size\n\n", ESP.getHeapSize());
  Serial.printf("%u free heap size\n\n", ESP.getFreeHeap());
  portEXIT_CRITICAL(&mux);
}

void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;  // wait for Serial port to be opened
  Serial.printf("ESP32 Start\n");

  // turn off WiFi
  WiFi.mode(WIFI_OFF);
  Serial.printf("WiFi turn off\n\n");

  // start timer on report esp memory status
  timer = timerBegin(0, 80, true);
  timerAttachInterrupt(timer, &onTimerStatusReport, true);
  timerAlarmWrite(timer, 10000000, true);
  timerAlarmEnable(timer);

  // initial model
  model.add(new framework::input({ 1, 3, 16, 16 }));
  model.add(new framework::conv2d("Conv2d_1", model.back(), 1, 3, 1, 1, true));

  model.summary();

  // initial weight
  std::fill_n(data, (1 * 3 * 16 * 16), 1.0f);
  std::fill_n(conv2d_w, ((3 * 3 * 3 * 1) + 1), 1.0f);
  weights.push_back(conv2d_w);

  model.loadWeight(weights);

  // Tasks
  xTaskCreatePinnedToCore(
    taskFramework,
    "AI Framework Task",
    40000,  // Usually use 1000 (in ESP32 it's bytes)
    NULL,
    1,
    NULL,  // Task handler
    1
  );
}

void loop() {

}
