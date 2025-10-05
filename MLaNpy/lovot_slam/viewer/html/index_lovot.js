let slamview = {};

slamview.lovotPose = {
    'x': 0, 'y': 0,
    'yaw': 0,
    'covariantMatrix': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'reliability': null
};

slamview.markerPoses = {};

slamview.MAP_VIZ_SCALE = 2.0;
slamview.MAP_RESOLUTION = 0.05;
slamview.mapImage = document.getElementById("map-img");
slamview.mapHeight = slamview.mapImage.clientHeight
slamview.mapWidth = slamview.mapImage.clientWidth

// map origin written by lovot_webview.py
{{ mapOrigin }}

slamview.mapCanvas = document.getElementById("map-canvas");
slamview.mapContext = slamview.mapCanvas.getContext("2d");

slamview.client = new LocalizationClient(location.hostname);

slamview.webSocket = new WebSocket("ws://" + location.hostname + ":38001");

slamview.covariancePool = ["slam:pose:covariance", "slam:pose_visual:covariance", "slam:pose:amcl"];

// Initialize marker action client
slamview.markerActionClient = new MarkerActionClient(location.hostname);

getUnixTime = function () {
    return Math.round((new Date()).getTime() / 1000);
};

slamview.webSocket.onmessage = function (event) {
    // console.log(event.data);
    result = JSON.parse(event.data);
    // console.log(result.result);
    if (result.result) {
        slamview.drawAll();
        if ('translation' in result.result) {
            slamview.lovotPose.x = parseInt(result.result.translation.x / slamview.MAP_RESOLUTION * slamview.MAP_VIZ_SCALE);
            slamview.lovotPose.y = parseInt(slamview.mapHeight - result.result.translation.y / slamview.MAP_RESOLUTION * slamview.MAP_VIZ_SCALE);
            slamview.lovotPose.yaw = slamview.getYawAngleFromQuaternion(result.result.rotation);
        }
        else if ('slam:pose:covariance' in result.result) {
            slamview.lovotPose.covariantMatrix = result.result["slam:pose:covariance"][1].split(",").map(Number);
        }
        else if ('slam:failure_detection:result' in result.result) {
            const keys = ["timestamp", "reliability", "detection", "likelihood"];
            const values = result.result["slam:failure_detection:result"].map(Number);
            if (values[0] === 0) return;
            // update only if timestamp is valid
            slamview.lovotPose.reliability = {};
            keys.forEach((key, i) => slamview.lovotPose.reliability[key] = values[i]);
            slamview.lovotPose.reliability["localtime"] = getUnixTime();
        }
        else if ('slam:markers_position' in result.result) {
            // key-value pair, ID: pose (4x4 matrix)
            const markerPoses = result.result["slam:markers_position"];
            slamview.markerPoses = {};
            for (let markerId in markerPoses) {
                const markerPose = JSON.parse(markerPoses[markerId]);
                const z_angle_on_map = Math.atan2(markerPose[1][2], markerPose[0][2]);
                slamview.markerPoses[markerId] = {
                    x: parseInt((markerPose[0][3] - slamview.mapOrigin[0]) / slamview.MAP_RESOLUTION * slamview.MAP_VIZ_SCALE),
                    y: parseInt(slamview.mapHeight - (markerPose[1][3] - slamview.mapOrigin[1]) / slamview.MAP_RESOLUTION * slamview.MAP_VIZ_SCALE),
                    theta: z_angle_on_map,
                    name: "marker_" + markerId
                }
            }
        }

        const exists = slamview.covariancePool.some(key => key in result.result);
        if (exists) {
            update_lovot_data(result);
        }

        if (Array.isArray(result.result)) {
            if (result.result.includes('visual') || result.result.includes('depth')) {
                const localizerMode = result.result[0];
                if (localizerMode === 'visual') {
                    changeColor("cov-visual", "green");
                    changeColor("cov-depth", "gray");
                }
                else {
                    changeColor("cov-visual", "gray");
                    changeColor("cov-depth", "green");
                }
            }
        }
    }
};

function update_lovot_data(result) {
    const formattingMatrix = (matrix) => {
        if (!Array.isArray(matrix) || matrix.length !== 9) {
            throw new Error("pCovarianceMatrix must be an array with 9 elements.");
        }

        const reshapedMatrix = [];
        for (let i = 0; i < 3; i++) {
            reshapedMatrix.push(matrix.slice(i * 3, (i + 1) * 3));
        }

        return reshapedMatrix;
    }

    const wrapMatrixasPre = (matrix) => {
        const preElement = document.createElement("pre");
        preElement.style.margin = "0";
        preElement.textContent = "[\n" + matrix.map(row => " " + JSON.stringify(row)).join(",\n") + "\n]";
        return preElement;
    }

    // Update Covariance value
    if ('slam:pose:covariance' in result.result) {
        const covarianceCell = document.getElementById("covariance-value");
        const covarianceEvalCell = document.getElementById("covariance-evl");

        let pCovarianceMatrix = result.result["slam:pose:covariance"][1].split(',').map(Number);
        pCovarianceMatrix = formattingMatrix(pCovarianceMatrix);

        covarianceCell.textContent = "";
        const el_matrix = wrapMatrixasPre(pCovarianceMatrix);
        covarianceCell.appendChild(el_matrix);

        const evaluation = evaluate_covariance(pCovarianceMatrix);
        covarianceEvalCell.textContent = evaluation;
    }

    // Update Visual Covariance value
    if ('slam:pose_visual:covariance' in result.result) {
        const visualCovarianceCell = document.getElementById("visual-covariance-value");
        const visualCovarianceEvalCell = document.getElementById("visual-covariance-evl");

        let pvCovarianceMatrix = result.result["slam:pose_visual:covariance"][1].split(',').map(Number);
        pvCovarianceMatrix = formattingMatrix(pvCovarianceMatrix);

        visualCovarianceCell.textContent = "";
        const el_matrix = wrapMatrixasPre(pvCovarianceMatrix);
        visualCovarianceCell.appendChild(el_matrix);

        const evaluation = evaluate_covariance(pvCovarianceMatrix);
        visualCovarianceEvalCell.textContent = evaluation;
    }

    // Update AMCL Pose value (no evaluation)
    if ('slam:pose:amcl' in result.result) {
        const amclPoseCell = document.getElementById("amcl-pose-value");
        const amclPoseEvalCell = document.getElementById("amcl-pose-evl");

        let amclPose = result.result["slam:pose:amcl"][1].split(',').map(Number);
        amclPoseCell.textContent = amclPose;

        amclPoseEvalCell.textContent = "N/A";
    }
}

function evaluate_covariance(covariance) {
    if (typeof math === "undefined") {
        throw new Error("This function requires math.js for matrix computations.");
    }

    const covarianceFlat = covariance.flat();
    const size = Math.sqrt(covarianceFlat.length);
    if (!Number.isInteger(size)) {
        throw new Error("Input covariance array is not a valid flattened square matrix.");
    }

    const reshapedCovariance = [];
    for (let i = 0; i < size; i++) {
        reshapedCovariance.push(covarianceFlat.slice(i * size, (i + 1) * size));
    }

    // get ellipse data
    const subMatrix = [
        [reshapedCovariance[0][0], reshapedCovariance[0][1]],
        [reshapedCovariance[1][0], reshapedCovariance[1][1]]
    ];

    const symPositionCovariance = math.divide(
        math.add(subMatrix, math.transpose(subMatrix)),
        2
    );

    const { values: eigenValues, vectors: eigenVectors } = math.eigs(symPositionCovariance);
    const angle = Math.atan2(eigenVectors[1][0], eigenVectors[0][0]);
    const clippedValues = eigenValues.map((val) => Math.max(0.0, Math.min(val, 100.0)));
    const axes = clippedValues.map(Math.sqrt);

    // get angle sigma
    if (reshapedCovariance.length !== 3 || reshapedCovariance[0].length !== 3) {
        throw new Error("The covariance matrix must be 3x3.");
    }
    const angleSigma = reshapedCovariance[2][2];


    // define thresholds
    const NAVIGATABLE_POSITIONAL_ERROR = 0.75;

    const maxAxis = Math.max(...axes);
    return maxAxis > NAVIGATABLE_POSITIONAL_ERROR ? "high" : "low";
}

function changeColor(elementId, color) {
    document.getElementById(elementId).style.color = color;
}

slamview.webSocket.onopen = function (event) {
    console.log("websocket connected");
};

slamview.getYawAngleFromQuaternion = function (rotation) {
    return Math.atan2(2 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1 - 2 * (rotation.y ** 2 + rotation.z ** 2))
};

slamview.drawCircle = function (x, y, color = "red") {
    slamview.mapContext.beginPath();
    slamview.mapContext.arc(x, y, 5, 0, 2 * Math.PI);
    slamview.mapContext.fillStyle = color;
    slamview.mapContext.globalAlpha = 0.8;
    slamview.mapContext.fill();
};

slamview.drawConfidenceEllipse = function (x, y, covarianceMatrix, color = "green") {
    // convert 3x3 matrix into 2x2 matrix
    const H = [[covarianceMatrix[0], covarianceMatrix[1]], [covarianceMatrix[3], covarianceMatrix[4]]]
    const ans = math.eigs(H)
    const E = ans.values
    const U = ans.vectors

    slamview.mapContext.beginPath();
    slamview.mapContext.ellipse(x, y,
        Math.sqrt(E[0]) / slamview.MAP_RESOLUTION * slamview.MAP_VIZ_SCALE,
        Math.sqrt(E[1]) / slamview.MAP_RESOLUTION * slamview.MAP_VIZ_SCALE,
        Math.atan2(U[0][1], U[0][0]), 0, 2 * Math.PI);
    slamview.mapContext.stroke();
    slamview.mapContext.fillStyle = color;
    slamview.mapContext.globalAlpha = 0.4;
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
    } else if (name === "marker") {
        color = "darkorange";
    }
    slamview.drawCircle(x, y, color);

    // Draw arrow
    ax = Math.round(arrowSize * Math.cos(theta));
    ay = Math.round(arrowSize * Math.sin(theta));
    slamview.drawArrow(x, y, x + ax, y - ay, color);
};

slamview.drawLovotPose = function (x, y, theta, confidenceMatrix, arrowSize = 25, arrowOffset = 10) {
    slamview.drawConfidenceEllipse(x, y, confidenceMatrix);
    slamview.drawPose(x, y, theta, name, arrowSize, arrowOffset);
};

slamview.drawReliabilityText = function () {
    const rel = slamview.lovotPose.reliability;
    if (rel) {
        const currenttime = getUnixTime();
        slamview.mapContext.fillStyle = "rgba(" + [255, 255, 255, 0.5] + ")";
        slamview.mapContext.fillRect(8, 8, 300, 16);
        slamview.mapContext.font = "14px Arial";
        if (currenttime - rel["localtime"] > 30) {
            slamview.mapContext.fillStyle = "gray";  // old data
        } else {
            slamview.mapContext.fillStyle = "blue";
        }
        slamview.mapContext.fillText(
            `reliability: ${(rel.reliability).toFixed(2)}, detection: ${(rel.detection).toFixed(2)}, likelihood: ${(rel.likelihood).toFixed(2)}`,
            10, 22);
    }
};

slamview.drawAll = function () {
    slamview.mapContext.clearRect(0, 0, slamview.mapCanvas.clientWidth, slamview.mapCanvas.clientHeight);
    slamview.drawLovotPose(slamview.lovotPose.x, slamview.lovotPose.y, slamview.lovotPose.yaw,
        slamview.lovotPose.covariantMatrix);
    for (let key in slamview.markerPoses) {
        const marker = slamview.markerPoses[key];
        slamview.drawPose(marker.x, marker.y, marker.theta, "marker", 15);
    }
    {{ drawPoses }}
    {{ drawUnwelcomedArea }}
    slamview.drawReliabilityText();
};

slamview.getLovotPose = function () {
    var query = {
        cmd: "?TF,GET",
        qid: "123",
        attrs: {
            parent_frame_id: "omni_image_origin",
            child_frame_id: "base",
            timestamp: 0
        }
    }
    var json = JSON.stringify(query);
    slamview.webSocket.send(json);
};

slamview.getLovotPoseConfidence = function () {
    var query = {
        cmd: "?STM,HMGET",
        qid: "123",
        attrs: {
            keys: ["slam:pose:covariance"],
            fields: ["timestamp", "covariance"]
        }
    }
    var json = JSON.stringify(query);
    slamview.webSocket.send(json);
};

slamview.getLovotPoseVisualCovariance = function () {
    var query = {
        cmd: "?STM,HMGET",
        qid: "123",
        attrs: {
            keys: ["slam:pose_visual:covariance"],
            fields: ["timestamp", "covariance"]
        }
    }
    var json = JSON.stringify(query);
    slamview.webSocket.send(json);
};

slamview.getLovotPoseAMCL = function () {
    var query = {
        cmd: "?STM,HMGET",
        qid: "123",
        attrs: {
            keys: ["slam:pose:amcl"],
            fields: ["timestamp", "pose", "covariance"]
        }
    }
    var json = JSON.stringify(query);
    slamview.webSocket.send(json);
};

// Not support GET??
slamview.getLovotPoseLocalizer = function() {
    var query = {
        cmd: "?STM,MGET",
        qid: "123",
        attrs: {
          keys: ["slam:pose:localizer"]
        }
    }
    var json = JSON.stringify(query);
    slamview.webSocket.send(json);
};

slamview.getLocalizationReliability = function () {
    var query = {
        cmd: "?STM,HMGET",
        qid: "123",
        attrs: {
            keys: ["slam:failure_detection:result"],
            fields: ["timestamp", "reliability", "detection", "likelihood"]
        }
    }
    var json = JSON.stringify(query);
    slamview.webSocket.send(json);
};

slamview.getMarkerPoses = function () {
    var query = {
        cmd: "?LTM,HGETALL",
        qid: "123",
        attrs: {
            keys: ["slam:markers_position"]
        }
    }
    var json = JSON.stringify(query);
    slamview.webSocket.send(json);
}

slamview.mainLoop = function () {
    const lovot_pose_update_task = setInterval(slamview.getLovotPose, 100);
    // localcation accuracy data
    const lovot_pose_confidence_update_task = setInterval(slamview.getLovotPoseConfidence, 1000);
    const lovot_pose_visual_covariance_update_task = setInterval(slamview.getLovotPoseVisualCovariance, 1000);
    const lovot_pose_amcl_update_task = setInterval(slamview.getLovotPoseAMCL, 1000);
    const lovot_pose_localizer_update_task = setInterval(slamview.getLovotPoseLocalizer, 1000);
    const marker_poses_update_task = setInterval(slamview.getMarkerPoses, 1000);

    const localization_reliability_update_task = setInterval(slamview.getLocalizationReliability, 2000);
    const draw_task = setInterval(slamview.drawAll, 100);
};

window.onload = function () {
    slamview.mainLoop();
    console.log("window loaded");
    slamview.drawAll();
};
