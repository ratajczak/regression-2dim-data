var convnetjs = require('convnetjs');

// create a net out of it
var net = new convnetjs.Net();
var d = 12;
var layer_defs = [];
layer_defs.push({type:'input', out_sx:2, out_sy:1, out_depth:d});
layer_defs.push({type:'fc', num_neurons:30, activation:'sigmoid'});
layer_defs.push({type:'regression', num_neurons:2});
var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// var my_data = [
//     0,1,2,3,4,5,6,
//     0,1,2,3,4,5,6,
//     0,1,2,3,4,5,6
// ];

var my_data = [
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
  [0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9]
];



var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, momentum:0.2, batch_size:1, l2_decay:0.001});

var learn = function () {
  for(var j = 0; j < 100; j++){
    for (var i = 0; i < my_data.length - d; i++) {
        var data = my_data.slice(i, i + d);
        var real_value = [my_data[i + d]][0];
        var x = new convnetjs.Vol(2, 1, d, 0);

        for (var ii = 0; ii < d; ii++) {
          x.set(0,0,ii,data[ii][0]);
          x.set(1,0,ii,data[ii][1]);
        }


        trainer.train(x, real_value);
        var predicted_values = net.forward(x);
        console.log("data: [" + data + "] -> value: " + real_value);
        console.log("prediction in learn stage is: " + predicted_values.w[0] + ', ' + predicted_values.w[1]);
    }
  }

}

var predict = function(data){
  var x = new convnetjs.Vol(2, 1, d, 0);

  for (var ii = 0; ii < d; ii++) {
    x.set(0,0,ii,data[ii][0]);
    x.set(1,0,ii,data[ii][1]);
  }

  var predicted_value = net.forward(x);
  return predicted_value.w ;
}

learn();
var item = [[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[0,3],[1,4],[2,5],[3,6],[4,7]];
var predicted = predict(item);
console.log("predicted value for [" + item + "] is: " + predicted[0] + ', ' + predicted[1]);
var item = [[6,9],[0,3],[1,4],[2,5],[3,6]];
var predicted = predict(item);
console.log("predicted value for [" + item + "] is: " + predicted[0] + ', ' + predicted[1]);
