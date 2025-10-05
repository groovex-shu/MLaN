function generateUuid() {
    // https://github.com/GoogleChrome/chrome-platform-analytics/blob/master/src/internal/identifier.js
    // const FORMAT: string = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";
    let chars = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".split("");
    for (let i = 0, len = chars.length; i < len; i++) {
        switch (chars[i]) {
            case "x":
                chars[i] = Math.floor(Math.random() * 16).toString(16);
                break;
            case "y":
                chars[i] = (Math.floor(Math.random() * 4) + 8).toString(16);
                break;
        }
    }
    return chars.join("");
}

class LocalizationClient {
    constructor(host) {
        this.host = host;

        this.publishedCommands = [];

        this.ws = new WebSocket("ws://" + host + ":38001");
        this.block = false;

        this.ws.onmessage = (event) => {
            const result = JSON.parse(event.data);
            console.log(result.result);
            if (result.result) {
                // response is space separated string: "command uuid messages"
                const responses = result.result.split(" ");
                if (responses.length < 2) {
                    console.error("unexpected response: " + result.result);
                    return;
                }

                // alert only the responses of the published commands
                const qid = responses[1];
                delete responses[1];
                if (!(qid in this.publishedCommands)) return;
                delete this.publishedCommands[qid];
                this.block = false;

                window.location.reload();
            }
        };

        this.ws.onopen = (event) => {
            console.log("localization client websocket connected");
            this.subscribeResponse();
        };
    }

    subscribeResponse() {
        const query = {
            cmd: "@STM,SUBSCRIBE",
            qid: "123",
            attrs: {
                channel: "slam:response"
            }
        }
        const json = JSON.stringify(query);
        this.ws.send(json);
    }

    publishCommand(command, args) {
        if (this.block) {
            console.warn("failed to publish a new command, since it is waiting for another response");
            return;
        }
        this.block = true;
        const qid = generateUuid();
        let value = command + " " + qid;
        if (args.length > 0) value += (" " + args.join(" "));
        const query = {
            cmd: "!STM,PUBLISH",
            qid: "123",
            attrs: {
                channel: "slam:command",
                value: value
            }
        }
        this.publishedCommands[qid] = value;
        const json = JSON.stringify(query);
        this.ws.send(json);
        console.info("command sent: " + json);

        setTimeout(() => {
            if (this.block) {
                this.block = false;
                console.warn("command response timeout occured");
            }
        }, 15 * 1000);
    }

    changeMap() {
        this.publishCommand('change_map', ['latest']);
    }

    undeployMap() {
        this.publishCommand('undeploy_map', []);
    }
}

class MarkerActionClient {
    constructor(host) {
        this.host = host;
        this.ws = new WebSocket("ws://" + host + ":38001");
        this.publishedCommands = [];

        this.ws.onmessage = (event) => {
            const result = JSON.parse(event.data);
            if (result.result) {
                const markerAction = JSON.parse(result.result);
                // console.log(markerAction);
                this.updateMarkerActionUI(markerAction);
            }
        };

        this.ws.onopen = (event) => {
            console.log("marker action client websocket connected");
            this.subscribeResponse();
        };
    }

    subscribeResponse() {
        const query = {
            cmd: "@STM,SUBSCRIBE",
            qid: "123",
            attrs: {
                channel: "slam:marker_node_action"
            }
        }
        const json = JSON.stringify(query);
        this.ws.send(json);
    }

    updateMarkerActionUI(markerAction) {
        const actionCell = document.getElementById("marker-action-value");
        const timeCell = document.getElementById("marker-action-time");

        // Format the action display
        const actionText = `Action: ${markerAction.action}, Marker ID: ${markerAction.marker_id}`;
        actionCell.textContent = actionText;

        // Format the timestamp
        const date = new Date(markerAction.time * 1000);
        const timeText = date.toLocaleTimeString();
        timeCell.textContent = timeText;
    }
}
