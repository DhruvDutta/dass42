const qlist=["I found myself getting upset by quite trivial things.",
    "I was aware of dryness of my mouth.",
    "I couldn't seem to experience any positive feeling at all.",
    "I experienced breathing difficulty (eg, excessively rapid breathing, breathlessness in the absence of physical exertion).",
    "I just couldn't seem to get going.",
    "I tended to over-react to situations.",
    "I had a feeling of shakiness (eg, legs going to give way).",
    "I found it difficult to relax.",
    "I found myself in situations that made me so anxious I was most relieved when they ended.",
    "I felt that I had nothing to look forward to.",
    "I found myself getting upset rather easily.",
    "I felt that I was using a lot of nervous energy.",
    "I felt sad and depressed.",
    "I found myself getting impatient when I was delayed in any way (eg, elevators, traffic lights, being kept waiting).",
    "I had a feeling of faintness.",
    "I felt that I had lost interest in just about everything.",
    "I felt I wasn't worth much as a person.",
    "I felt that I was rather touchy.",
    "I perspired noticeably (eg, hands sweaty) in the absence of high temperatures or physical exertion.",
    "I felt scared without any good reason.",
    "I felt that life wasn't worthwhile.",
    "I found it hard to wind down.",
    "I had difficulty in swallowing.",
    "I couldn't seem to get any enjoyment out of the things I did.",
    "I was aware of the action of my heart in the absence of physical exertion (eg, sense of heart rate increase, heart missing a beat).",
    "I felt down-hearted and blue.",
    "I found that I was very irritable.",
    "I felt I was close to panic.",
    "I found it hard to calm down after something upset me.",
    "I feared that I would be \"thrown\" by some trivial but unfamiliar task.",
    "I was unable to become enthusiastic about anything.",
    "I found it difficult to tolerate interruptions to what I was doing.",
    "I was in a state of nervous tension.",
    "I felt I was pretty worthless.",
    "I was intolerant of anything that kept me from getting on with what I was doing.",
    "I felt terrified.",
    "I could see nothing in the future to be hopeful about.",
    "I felt that life was meaningless.",
    "I found myself getting agitated.",
    "I was worried about situations in which I might panic and make a fool of myself.",
    "I experienced trembling (eg, in the hands).",
    "I found it difficult to work up the initiative to do things."]

var curr_q=-1
var ans_list={}
function load_q(){
    if(curr_q>=qlist.length-1){
        submit();
        return
    }
    curr_q++;
    document.getElementById('question').innerText=qlist[curr_q];
    $(".progress-bar").animate({
        width: `${curr_q*100/(qlist.length)}%`
    }, 200);
}
load_q()

function ans(id){
    ans_list[`${curr_q}`]=`${id/4}`
    setTimeout(() => {
        document.getElementById('options').innerHTML = document.getElementById('options').innerHTML
    }, 150);
    //console.log(ans_list)
    load_q()
}
function submit(){
    document.getElementById('box').innerHTML += "<div class='fs-2 text-center' id='submit-text'>Submitting</div>"
    document.getElementById('form').remove()
    document.getElementById('progress').style.width=`100%`;

    $.ajax({
        type : "POST",
        url : "/",
        contentType: 'application/json',
        data : ans_list,
        success:function(response){
            response = JSON.parse(response)
            document.getElementById('result').classList.remove('d-none')
            document.getElementById('smile').classList.remove('d-none')
            document.getElementById('submit-text').remove()
            document.getElementById('dep').innerHTML=`${response['dep'][0]} <br> ${response['dep'][1]}`
            document.getElementById('strs').innerHTML=`${response['strs'][0]} <br> ${response['strs'][1]}`
            document.getElementById('anx').innerHTML=`${response['anx'][0]} <br> ${response['anx'][1]}`
        }
      });
}