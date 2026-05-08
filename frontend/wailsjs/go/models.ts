export namespace backend_golang {

	export class BatchCompletionFinishedEvent {
	    batchId: string;

	    static createFrom(source: any = {}) {
	        return new BatchCompletionFinishedEvent(source);
	    }

	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.batchId = source["batchId"];
	    }
	}
	export class BatchCompletionRequest {
	    url: string;
	    headers: {[key: string]: string};
	    count: number;
	    body: any;

	    static createFrom(source: any = {}) {
	        return new BatchCompletionRequest(source);
	    }

	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.url = source["url"];
	        this.headers = source["headers"];
	        this.count = source["count"];
	        this.body = source["body"];
	    }
	}
	export class BatchCompletionUpdate {
	    itemId: number;
	    status: string;
	    delta?: string;
	    text?: string;
	    error?: string;

	    static createFrom(source: any = {}) {
	        return new BatchCompletionUpdate(source);
	    }

	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.itemId = source["itemId"];
	        this.status = source["status"];
	        this.delta = source["delta"];
	        this.text = source["text"];
	        this.error = source["error"];
	    }
	}
	export class BatchCompletionUpdateEvent {
	    batchId: string;
	    updates: BatchCompletionUpdate[];

	    static createFrom(source: any = {}) {
	        return new BatchCompletionUpdateEvent(source);
	    }

	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.batchId = source["batchId"];
	        this.updates = this.convertValues(source["updates"], BatchCompletionUpdate);
	    }

		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class FileInfo {
	    name: string;
	    size: number;
	    isDir: boolean;
	    modTime: string;

	    static createFrom(source: any = {}) {
	        return new FileInfo(source);
	    }

	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.size = source["size"];
	        this.isDir = source["isDir"];
	        this.modTime = source["modTime"];
	    }
	}
	export class MIDIMessage {
	    messageType: string;
	    channel: number;
	    note: number;
	    velocity: number;
	    control: number;
	    value: number;

	    static createFrom(source: any = {}) {
	        return new MIDIMessage(source);
	    }

	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.messageType = source["messageType"];
	        this.channel = source["channel"];
	        this.note = source["note"];
	        this.velocity = source["velocity"];
	        this.control = source["control"];
	        this.value = source["value"];
	    }
	}

}
