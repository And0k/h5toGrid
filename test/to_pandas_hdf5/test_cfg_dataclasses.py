# -*- coding: utf-8 -*-
import pytest

from cfg_dataclasses import *


# @pytest.mark.skip(reason="passed")
@pytest.mark.parametrize('names', [
    'Get200HTTP_RespCode_Get_200_HTTP_RespCode'
    ])
def test_camel2snake(names):
    out = camel2snake(names)
    assert out == 'get200http_resp_code_get_200_http_resp_code'