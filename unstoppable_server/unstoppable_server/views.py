from django.http import HttpResponse
import os
import time
from django.shortcuts import render, render_to_response
from django.views.decorators.csrf import csrf_exempt
import json
from .one_click_helper import get_one_click_url
from .controllers import get_food_information_by_path


def index(request):
    return render_to_response("upload.html")


def hello(request):
    return HttpResponse("Food is unstopped!")


@csrf_exempt
def upload_file(request):
    if request.method == "POST":
        uploaded_image = request.FILES.get("file")
        top_num = int(request.POST.get("top_num"))
        if uploaded_image is None:
            return HttpResponse("No uploaded image.")
        else:
            if not os.path.exists("uploaded_images"):
                os.mkdir("uploaded_images")

            file_name = "uploaded_images/%s.%s" % (time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())),
                                                   uploaded_image.content_type.split("/")[-1])

            with open(file_name, 'wb+') as f:
                for chunk in uploaded_image.chunks():
                    f.write(chunk)
            print("Get image from client and save to %s, excepted top-%d" % (file_name, top_num))
            food_info_list = get_food_information_by_path(file_name, top_num)
            if len(food_info_list) > 0:
                for food_info in food_info_list:
                    # print("Try to get info for %s" % food_info["name"])
                    food_url = food_info["url"]
                    # food_info["url"] = food_url
                    # print("Get url for %s successfully: %s" % (food_info["name"], food_url))
                food_json = json.dumps(food_info_list)
                return HttpResponse(food_json)
            else:
                return HttpResponse("{}")
    else:
        return render(request, "upload.html")
