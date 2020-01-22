from pyproj import Geod

class GetDistance :
    def get_distance(self, start, to):
        res = u'[ %(from)s ]から[ %(to)s ]まで %(distance_str)s (%(distance)s)'

        q = Geod(ellps='WGS84')
        fa, ba, d = q.inv(start['lon'], start['lat'], to['lon'], to['lat'])

        print( res % {
            'from'  : start['name'],
            'to'    : to['name'],
            'distance_str'  : self._cutdown(d),
            'distance'      : d,
        })

    def _cutdown(self, num):
        val = int(round(num))
        if val < 1000:
            return '%sm' % val
        else:
            km = val * 0.001
            return '%sKm' % round(km, 1)



if __name__ == '__main__':
    # 東京タワー
    tokyo_tw = {
        'name' : u'東京タワー',
        'lat'  : 35.65861,
        'lon'  :139.745447,
    }

    #芝公園
    shiba_park = {
        'name' :u'芝公園',
        'lat'  :35.654071,
        'lon'  :139.749838,
    }

    #富士山
    mt_fuji = {
        'name': u'富士山',
        'lat':35.360556,
        'lon':138.727778,
    }

    g = GetDistance()
    g.get_distance(tokyo_tw, shiba_park)
    g.get_distance(tokyo_tw, mt_fuji)