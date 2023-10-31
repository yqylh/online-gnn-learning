nodeNum = 231443
out = {}
for (var i = 0; i < nodeNum; i++) {
    out.push({
        "id": i,
        "timestamp": Math.floor(Math.random() * 19991107)
    })
    // out[i] = Math.floor(Math.random() * 19991107)
}
// console.log(JSON.stringify(out))
let fs = require('fs')
fs.writeFile('vertex_timestamp.json', JSON.stringify(out), function (err) {})