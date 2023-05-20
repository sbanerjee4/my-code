import java.util.*;
import java.io.*;
import java.lang.*;

public class fenceplan
{
   static ArrayList[] moo;
   static boolean[] visited;
   static cow[] cow;
   static long x1, x2, y1, y2;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("fenceplan.in"));
      int n = in.nextInt(), m = in.nextInt();
      moo = new ArrayList[n];
      cow = new cow[n];
      for(int i = 0; i < n; i++)
         cow[i] = new cow(in.nextInt(), in.nextInt());
      for(int i = 0; i < n; i++)
        moo[i] = new ArrayList<Integer>();
      for(int i = 0; i < m; i++)
      {
        int a = in.nextInt() - 1, b = in.nextInt() - 1;
        moo[a].add(b);
        moo[b].add(a);
      }
      in.close();
     
      visited = new boolean[n];
      long perim = Long.MAX_VALUE;
      for(int i = 0; i < n; i++)
      {
          if(!visited[i])
          {
               x1 = 100000001;
               x2 = -1;
               y1 = 100000001;
               y2 = -1;
               ff(i);
               perim = Math.min(perim, 2 * (x2 - x1) + 2 * (y2 - y1));
          }
      }
      
      long result = perim;
      PrintWriter out = new PrintWriter(new File("fenceplan.out"));
      System.out.println(result);
      out.println(result);
      out.close();
   }
   
   static void ff(int current)
   {
        if(visited[current])
           return;
        visited[current] = true;
        x1 = Math.min(x1, cow[current].x);
        x2 = Math.max(x2, cow[current].x);
        y1 = Math.min(y1, cow[current].y);
        y2 = Math.max(y2, cow[current].y);
        for(int i = 0; i < moo[current].size(); i++)
             ff((int)(moo[current].get(i)));
   }
   
   static class cow
   {
        public long x;
        public long y;
        public cow(long a, long b)
        {
            x = a;
            y = b;
        }
   }
}