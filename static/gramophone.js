/*First beam note*/

const t1 = gsap.timeline({delay:2, repeat:-1, paused:true});
t1.fromTo(".beam:nth-child(1)",{
  opacity:0
},
{
  x:-40,
  opacity:1, 
  duration:2
})
.to(".beam:nth-child(1)", {
  opacity:0, 
  duration:0.2
}, "-=0.1");

/*Minim note*/
const t2 = gsap.timeline({repeat:-1, delay:1, paused:true});
t2.fromTo(".minim", {
  opacity:0
}, {
  x:-20,
  opacity:1,
  duration:2
})
.to(".minim", {
  opacity:0,
  duration:0.2
}, "-=0.1");

/*Second beam note*/
const t3 = gsap.timeline({repeat:-1, delay:3, paused:true});
t3.fromTo(".beam2", {
  opacity:0
}, {
  opacity:1,
  x:-60,
  y:15,
  duration:4
});

setTimeout(playTimelines, 1000);
function playTimelines(){
  t1.play();
  t2.play();
  t3.play();
}