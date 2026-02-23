console.log("one"); // Synchronous
console.log("two"); // Synchronous

setTimeout(() => {
    console.log("three"); // Asynchronous
}, 1000);

console.log("four"); // Synchronous