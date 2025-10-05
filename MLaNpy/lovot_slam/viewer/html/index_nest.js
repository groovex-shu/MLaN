let slamview = {};

slamview.MAP_VIZ_SCALE = 2.0;
slamview.MAP_RESOLUTION = 0.05;
slamview.mapImage = document.getElementById("map-img");
slamview.mapHeight = slamview.mapImage.clientHeight
slamview.mapWidth = slamview.mapImage.clientWidth

slamview.mapCanvas = document.getElementById("map-canvas");
slamview.mapContext = slamview.mapCanvas.getContext("2d");

slamview.drawCircle = function (x, y, color = "red") {
    slamview.mapContext.beginPath();
    slamview.mapContext.arc(x, y, 5, 0, 2 * Math.PI);
    slamview.mapContext.fillStyle = color;
    slamview.mapContext.globalAlpha = 0.8;
    slamview.mapContext.fill();
};

slamview.drawArrow = function (fromx, fromy, tox, toy, color = "red") {
    var headlen = 10; // length of head in pixels
    var dx = tox - fromx;
    var dy = toy - fromy;
    var angle = Math.atan2(dy, dx);

    slamview.mapContext.beginPath();
    slamview.mapContext.strokeStyle = color;
    slamview.mapContext.moveTo(fromx, fromy);
    slamview.mapContext.lineTo(tox, toy);
    slamview.mapContext.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
    slamview.mapContext.moveTo(tox, toy);
    slamview.mapContext.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
    slamview.mapContext.stroke();
};

slamview.drawUnwelcomedArea = function (edges) {
    // only support 4-corner shape
    // edges should be 2d array, ex: [[0,0],[1,0],[1,1],[0,1]]
    slamview.mapContext.beginPath();
    slamview.mapContext.fillStyle = "orange";

    for (let i = 0; i < edges.length; i++) {
        if (i == 0) {
            slamview.mapContext.moveTo(edges[i][0], edges[i][1]);
        } else {
            slamview.mapContext.lineTo(edges[i][0], edges[i][1]);
        }
    }
    slamview.mapContext.lineTo(edges[0][0], edges[0][1]);
    slamview.mapContext.closePath();
    slamview.mapContext.globalAlpha = 0.8
    slamview.mapContext.fill();
};

slamview.drawPose = function (x, y, theta, name, arrowSize = 25, arrowOffset = 10) {
    // theta must be radian

    let color = "green";
    if (name === "entrance") {
        color = "red";
    } else if (name === "nest") {
        color = "blue";
    }
    slamview.drawCircle(x, y, color);

    // Draw arrow
    ax = Math.round(arrowSize * Math.cos(theta));
    ay = Math.round(arrowSize * Math.sin(theta));
    slamview.drawArrow(x, y, x + ax, y - ay, color);
};

{{ drawPoses }}
{{ drawUnwelcomedArea }}
