function custom_error(output_msg, title_msg)
{
    if (!title_msg)
        title_msg = 'Error';

    if (!output_msg)
        output_msg = 'No Message to Display.';

    custom_message(output_msg, title_msg, 'orange')
}

function custom_alert(output_msg, title_msg)
{
    if (!title_msg)
        title_msg = 'Alert';

    if (!output_msg)
        output_msg = 'No Message to Display.';

    custom_message(output_msg, title_msg, 'yellow')
}

function custom_message(output_msg, title_msg, color)
{
    var id = Math.random().toString(36).substr(2, 5);

    if (!title_msg)
        title_msg = 'Information';

    if (!color)
        color = 'theme';

    $("body").append($('<div id="customalert'+id+'" class="w3-modal" style="display:block"><div class="w3-modal-content w3-card-4">'+
    '<header class="w3-container w3-'+color+'"><span onclick="document.getElementById(\'customalert'+id+'\').remove()" class="w3-button w3-display-topright">&times;</span><h3>'+title_msg+'</h3></header>'+
    '<div class="w3-container"><p>'+output_msg+'</p></div></div></div>'));
}

delay = (function(){
    var timer = 0;
    return function(callback, ms){
        clearTimeout (timer);
        timer = setTimeout(callback, ms);
    };
})();
