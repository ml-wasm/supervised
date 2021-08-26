import initSupervised, { 
  initThreadPool,
  GaussianNaiveBayes,
  FloatsMatrix,
  FloatsVector 
} from '@ml.wasm/supervised';

(async () => {
  await initSupervised();
  await initThreadPool(navigator.hardwareConcurrency);

  const x = new FloatsMatrix([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]);
  const y = new FloatsVector([1, 1, 1, 2, 2, 2]);

  console.log(x.data);
  console.log(y.data);

  const clf = new GaussianNaiveBayes();
  clf.fit(x, y);

  console.log(clf.toString());

  const x1 = new FloatsMatrix([[-0.8, -1]]);
  const y1 = clf.predict(x1);
  console.log(y1.data);
})();
