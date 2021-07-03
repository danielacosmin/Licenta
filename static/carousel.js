const outtester = document.getElementsByClassName('scrolly');


function scrollification(mainConter ,setyup = {i, scrollcont, itemsScroll, customButtons, butLeft, butRight}){
    if(setyup.i == null){
        out = document.getElementsByClassName(mainConter)[0];
    }else{
        out = document.getElementsByClassName(mainConter)[setyup.i];
    }
    
    if(setyup.customButtons == true){
        l = out.querySelector(setyup.butLeft);
        r = out.querySelector(setyup.butRight);
    }

var out,
a = out.querySelector(setyup.scrollCont),l, r,
q =  out.querySelectorAll(setyup.itemsScroll);

$(l).click(function(){getBack();});

$(r).click(function(){getNext();});

function getNext(){
    for(var i = 0; i < q.length; i++){
        if(q[i].getBoundingClientRect().left <= a.getBoundingClientRect().left + 50){
            var index = q[i].offsetWidth;
            $(a).scrollLeft($(a).scrollLeft() + index);
        }
    }
}
function getBack(){
for(var i = 1; i < q.length; i++){
    if(q[i].getBoundingClientRect().left < a.getBoundingClientRect().left + 50){
        var index = q[i - 1].offsetWidth;
        $(a).scrollLeft($(a).scrollLeft() - index);}}}
}
for(var i = 0; i < outtester.length; i++){
    scrollification('scrolly', setyup = {
        i: i,
        scrollCont: '.carouselPre', 
        itemsScroll: '.itemsC',
        customButtons: true, 
        butLeft: '#left', 
        butRight: '#right'})
}

