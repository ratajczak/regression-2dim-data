var convnetjs = require('convnetjs');

// create a net out of it
var net = new convnetjs.Net();
var d = 5;
var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:d});
layer_defs.push({type:'fc', num_neurons:30, activation:'sigmoid'});
layer_defs.push({type:'regression', num_neurons:1});
var net = new convnetjs.Net();
net.makeLayers(layer_defs);

var my_data = [
    0,1,2,3,4,5,6,
    0,1,2,3,4,5,6,
    0,1,2,3,4,5,6
];



var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, momentum:0.2, batch_size:1, l2_decay:0.001});

var learn = function () {
  for(var j = 0; j < 500; j++){
    for (var i = 0; i < my_data.length - d; i++) {
        var data = my_data.slice(i, i + d);
        var real_value = [my_data[i + d]];
        var x = new convnetjs.Vol(data);
        trainer.train(x, real_value);
        var predicted_values = net.forward(x);
        console.log("data: [" + data + "] -> value: " + real_value);
        console.log("prediction in learn stage is: " + predicted_values.w[0]);
    }
  }

}

var predict = function(data){
  var x = new convnetjs.Vol(data);
  var predicted_value = net.forward(x);
  return predicted_value.w[0];
}

learn();
var item = [0,1,2,3,4];
console.log("predicted value for [" + item + "] is: " + predict(item));
var item = [4,5,6,0,1];
console.log("predicted value for [" + item + "] is: " + predict(item));
var item = [6,0,1,2,3];
console.log("predicted value for [" + item + "] is: " + predict(item));
