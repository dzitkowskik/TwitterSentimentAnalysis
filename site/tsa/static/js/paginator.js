function toPage(page)
{
    $('#id_page').val(page)
    $('#query_form').submit()
}

function toPrevious()
{
    if($('#id_page') > 1)
    {
        var page = $('#id_page').val() - 1
        $('#id_page').val(page)
        $('#query_form').submit()
    }
}

function toNext()
{
    var page = $('#id_page').val() + 1
    $('#id_page').val(page)
    $('#query_form').submit()
}

function hideBackArrowIfInPageOne()
{
    if($('#id_page').val() == 1)
    {
        $('#pagination_back').hide()
    }
}

$( document ).ready( hideBackArrowIfInPageOne );
