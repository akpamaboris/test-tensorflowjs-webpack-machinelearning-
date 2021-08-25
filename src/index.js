import * as tf from "@tensorflow/tfjs";

const x = tf.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
const y = tf.tensor([5.0, 7.0, 9.0, 11.0, 13.0, 15.0]);

const model = tf.sequential({
  layers: [
    tf.layers.dense({
      units: 1,
      inputShape: [1],
    }),
  ],
});

async function main() {
  await model.compile({
    optimizer: "sgd",
    loss: "meanSquaredError",
  });

  function onBatchEnd(batch, logs) {
    console.log(`Error: ${logs.loss}`);
  }

  await model.fit(x, y, {
    epochs: 500,
    verbose: true,
    callbacks: { onBatchEnd },
  });

  const prediction = await model.predict(tf.tensor([10]));
  const saveResult = await model.save("downloads://my-model");

  console.log(`Prediction: ${prediction}`);
}

main();
